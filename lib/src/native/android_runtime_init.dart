import 'dart:ffi';

import 'package:flutter/services.dart';

import '../android_runtime.dart';
import '../bindings/android_runtime_shim.dart';

const MethodChannel _channel = MethodChannel('flutter_litert');

Future<LiteRtRuntimeStatus>? _pending;

/// Why the last initialization left the CompiledModel runtime without a GPU,
/// kept so that later calls report it too.
String? _gpuError;

/// The LiteRT runtime this app was built with.
///
/// [LiteRtRuntime.playServices] only on Android apps built with
/// `flutterLitert.runtime=playServices`; [LiteRtRuntime.bundled] otherwise.
LiteRtRuntime get androidRuntime => usesPlayServicesRuntime
    ? LiteRtRuntime.playServices
    : LiteRtRuntime.bundled;

/// Initializes the Google Play services LiteRT runtime.
///
/// Apps built with `flutterLitert.runtime=playServices` must await this, from
/// the main isolate, before creating an `Interpreter` or a `CompiledModel`.
/// Isolates spawned afterwards can use both without calling it again. With the
/// bundled runtime, and on every platform but Android, it returns
/// [LiteRtRuntimeStatus.bundled] immediately.
///
/// [enableGpu] asks Play services for its GPU accelerator, for
/// `CompiledModel` with `Accelerator.gpu`. If that fails, the CompiledModel
/// runtime is initialized for CPU only and [LiteRtRuntimeStatus.gpuError]
/// says why. The first successful initialization fixes this for the process:
/// asking for the GPU later does not add it.
///
/// [enableAutomaticDownload] lets Play services download the runtime module
/// when the device does not have it yet. Without it, initialization fails
/// until the module is installed some other way.
///
/// Concurrent calls share one initialization. A runtime half that failed is
/// retried by the next call. Throws [LiteRtRuntimeUnavailableException] when
/// neither half could be initialized; when only one could, the returned status
/// says which and why.
Future<LiteRtRuntimeStatus> initializeAndroidRuntime({
  bool enableGpu = false,
  bool enableAutomaticDownload = true,
}) {
  final shim = androidRuntimeShim;
  if (shim == null || !shim.isPlayServices) {
    return Future.value(const LiteRtRuntimeStatus.bundled());
  }
  if (shim.interpreterState == kRuntimeStateReady &&
      shim.compiledModelState == kRuntimeStateReady) {
    return Future.value(_statusOf(shim, gpuError: _gpuError));
  }
  return _pending ??=
      _initialize(
        shim,
        enableGpu: enableGpu,
        enableAutomaticDownload: enableAutomaticDownload,
      ).whenComplete(() {
        _pending = null;
      });
}

Future<LiteRtRuntimeStatus> _initialize(
  AndroidRuntimeShim shim, {
  required bool enableGpu,
  required bool enableAutomaticDownload,
}) async {
  final reply = await _channel.invokeMapMethod<String, Object?>(
    'initializeRuntime',
    {
      'enableGpu': enableGpu,
      'enableAutomaticDownload': enableAutomaticDownload,
    },
  );
  // The Java and native halves are built from the same Gradle property, so a
  // mismatch means a broken build rather than a runtime condition.
  if (reply == null || reply['mode'] != 'playServices') {
    throw StateError(
      'flutter_litert: the Android plugin reported runtime '
      '${reply?['mode']}, but libflutter_litert_runtime.so was built for '
      'playServices. Rebuild the app.',
    );
  }
  var gpuError = reply['gpuError'] as String?;
  // Play services can initialize successfully with GPU support requested and
  // still provide no GPU accelerator, as on emulators.
  if (enableGpu &&
      gpuError == null &&
      shim.compiledModelState == kRuntimeStateReady &&
      shim.gpuAccelerator == nullptr) {
    gpuError =
        'Google Play services initialized LiteRT without a GPU accelerator '
        'on this device';
  }
  if (shim.compiledModelState == kRuntimeStateReady) _gpuError = gpuError;
  final status = _statusOf(
    shim,
    interpreterError: reply['interpreterError'] as String?,
    compiledModelError: reply['compiledModelError'] as String?,
    gpuError: gpuError,
  );
  if (!status.interpreterReady && !status.compiledModelReady) {
    throw LiteRtRuntimeUnavailableException(status);
  }
  return status;
}

/// Readiness is read from the shim rather than the channel reply, because the
/// shim is what the FFI bindings check.
LiteRtRuntimeStatus _statusOf(
  AndroidRuntimeShim shim, {
  String? interpreterError,
  String? compiledModelError,
  String? gpuError,
}) {
  final interpreterReady = shim.interpreterState == kRuntimeStateReady;
  final compiledModelReady = shim.compiledModelState == kRuntimeStateReady;
  return LiteRtRuntimeStatus(
    runtime: LiteRtRuntime.playServices,
    interpreterReady: interpreterReady,
    compiledModelReady: compiledModelReady,
    gpuAvailable: compiledModelReady ? shim.gpuAccelerator != nullptr : null,
    interpreterError: interpreterReady
        ? null
        : interpreterError ?? shim.interpreterError,
    compiledModelError: compiledModelReady
        ? null
        : compiledModelError ?? shim.compiledModelError,
    gpuError: gpuError,
  );
}
