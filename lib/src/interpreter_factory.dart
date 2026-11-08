// This factory is the supported way to get delegate-backed Interpreter
// inference: it selects the platform delegate (XNNPACK, Metal, CoreML) and
// falls back when one declines a model.
//
// PerformanceConfig.gpu() means the Metal delegate on iOS and macOS. Android
// has no Interpreter GPU delegate since 4.0.0, because LiteRT 2.x ships none:
// the bundled runtime warns once and uses XNNPACK, and Play services keeps its
// own CPU fallback. Android GPU inference goes through CompiledModel with
// Accelerator.gpu.
import 'dart:io';
import 'dart:math' as math;

import 'bindings/android_runtime_shim.dart';
import 'native/interpreter.dart';
import 'native/interpreter_options.dart';
import 'native/delegate.dart';
import 'native/isolate_interpreter.dart';
import 'delegates/xnnpack_delegate_native.dart';
import 'delegates/metal_delegate_native.dart';
import 'delegates/coreml_delegate_native.dart';
import 'performance_config.dart';

/// Factory for creating interpreter options with the package's platform
/// delegate mapping.
///
/// Supports Android, iOS, macOS, Windows, and Linux only. Import
/// `package:flutter_litert/native.dart` from native-only code to use this API.
/// For native/web acceleration selection, use
/// `CompiledModel.fromBufferWithConfigAsync` with `CompiledModelConfig.auto()`.
class InterpreterFactory {
  static void _warnDelegateFallback(String delegateName, Object error) {
    stderr.writeln(
      'flutter_litert: failed to initialize the $delegateName delegate; '
      'falling back to CPU. Error: $error',
    );
  }

  static bool _warnedAndroidGpuRemoved = false;

  /// Says once that `PerformanceConfig.gpu()` became XNNPACK on Android.
  ///
  /// Apps often build an interpreter per frame or per model, so a warning on
  /// every call would flood the log. Dart keeps statics per isolate, so a
  /// background isolate that also asks for the GPU prints it once more.
  static void _warnAndroidGpuRemovedOnce() {
    if (_warnedAndroidGpuRemoved) return;
    _warnedAndroidGpuRemoved = true;
    stderr.writeln(
      'flutter_litert: PerformanceConfig.gpu() uses XNNPACK on Android. The '
      'Android Interpreter GPU delegate (GpuDelegateV2) was removed in '
      'flutter_litert 4.0.0, because LiteRT 2.x has none. For the GPU, use '
      'CompiledModel with Accelerator.gpu.',
    );
  }

  /// Creates [InterpreterOptions] and an optional [Delegate] based on [config].
  ///
  /// Never throws for an unavailable delegate: it warns on stderr and returns
  /// options for the next best path. [PerformanceMode.gpu] on bundled Android
  /// returns an XNNPACK delegate, and on Play services no delegate.
  static (InterpreterOptions, Delegate?) create(
    PerformanceConfig? config, {
    bool addMediaPipeCustomOps = false,
  }) {
    final options = InterpreterOptions();
    if (addMediaPipeCustomOps) options.addMediaPipeCustomOps();
    final effectiveConfig = config ?? const PerformanceConfig();
    final threadCount =
        effectiveConfig.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);
    options.threads = threadCount;

    if (effectiveConfig.mode == PerformanceMode.disabled) {
      return (options, null);
    }
    if (effectiveConfig.mode == PerformanceMode.auto) {
      return _createAutoMode(options, threadCount);
    }
    if (effectiveConfig.mode == PerformanceMode.xnnpack) {
      return _createXnnpack(options, threadCount);
    }
    if (effectiveConfig.mode == PerformanceMode.gpu) {
      return _createGpu(options, threadCount);
    }
    if (effectiveConfig.mode == PerformanceMode.coreml) {
      return _createCoreml(options);
    }
    return (options, null);
  }

  /// Creates an [IsolateInterpreter] if conditions allow isolate-based inference.
  ///
  /// Returns null if [useIsolateInterpreter] is false, interpreter creation
  /// successfully applied a delegate, or the platform is macOS (where isolate
  /// sharing is unstable).
  static Future<IsolateInterpreter?> createIsolateIfNeeded(
    Interpreter interpreter,
    Delegate? delegate, {
    bool useIsolateInterpreter = true,
  }) async {
    if (!useIsolateInterpreter) return null;
    if (interpreter.hasActiveDelegate) return null;
    // Sharing a native interpreter across isolates is unstable on macOS.
    if (Platform.isMacOS) return null;
    return IsolateInterpreter.create(address: interpreter.address);
  }

  static (InterpreterOptions, Delegate?) _createAutoMode(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (Platform.isIOS) {
      return _createGpu(options, threadCount);
    }
    // Android, macOS, Linux, Windows, all use XNNPACK.
    return _createXnnpack(options, threadCount);
  }

  static (InterpreterOptions, Delegate?) _createXnnpack(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (!Platform.isAndroid &&
        !Platform.isIOS &&
        !Platform.isMacOS &&
        !Platform.isLinux &&
        !Platform.isWindows) {
      return (options, null);
    }
    // The Play services runtime exports no XNNPACK delegate API and applies
    // XNNPACK itself by default, so the request is already met.
    if (usesPlayServicesRuntime) {
      return (options, null);
    }
    try {
      final xnnOpts = XNNPackDelegateOptions(numThreads: threadCount);
      try {
        final xnnpackDelegate = XNNPackDelegate(options: xnnOpts);
        options.addDelegate(xnnpackDelegate);
        return (options, xnnpackDelegate);
      } finally {
        xnnOpts.delete();
      }
    } catch (error) {
      _warnDelegateFallback('XNNPACK', error);
      return (options, null);
    }
  }

  static (InterpreterOptions, Delegate?) _createGpu(
    InterpreterOptions options,
    int threadCount,
  ) {
    if (!Platform.isIOS && !Platform.isMacOS && !Platform.isAndroid) {
      return (options, null);
    }
    // The Play services runtime has no Interpreter GPU delegate. Keep the
    // CPU fallback every other delegate failure gets, but say why and where
    // its GPU is.
    if (usesPlayServicesRuntime) {
      _warnDelegateFallback(
        'GPU',
        playServicesUnsupported(
          'The Interpreter GPU delegate',
          'Use CompiledModel with Accelerator.gpu for the Play services GPU.',
        ),
      );
      return (options, null);
    }
    // The bundled Android runtime has no Interpreter GPU delegate either. Its
    // GPU is behind CompiledModel, so say so and use the fastest thing left:
    // XNNPACK is often several times faster than bare CPU.
    if (Platform.isAndroid) {
      _warnAndroidGpuRemovedOnce();
      return _createXnnpack(options, threadCount);
    }
    try {
      final gpuDelegate = GpuDelegate();
      options.addDelegate(gpuDelegate);
      return (options, gpuDelegate);
    } catch (error) {
      _warnDelegateFallback('GPU', error);
      return (options, null);
    }
  }

  static (InterpreterOptions, Delegate?) _createCoreml(
    InterpreterOptions options,
  ) {
    if (!Platform.isIOS && !Platform.isMacOS) {
      return (options, null);
    }
    try {
      final coreOpts = CoreMlDelegateOptions(enabledDevices: 1);
      try {
        final coremlDelegate = CoreMlDelegate(options: coreOpts);
        options.addDelegate(coremlDelegate);
        return (options, coremlDelegate);
      } finally {
        coreOpts.delete();
      }
    } catch (error) {
      _warnDelegateFallback('CoreML', error);
      return (options, null);
    }
  }
}
