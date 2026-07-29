/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show visibleForTesting;

import '../util/async_lock.dart';
import '../web/js_interop/litertjs_bindings.dart' as lrt;
import '../web/litertjs_interpreter.dart' show LiteRtRuntimeError;
import 'compiled_model_types.dart';

export 'compiled_model_types.dart';

/// LiteRT Next CompiledModel inference API, backed by LiteRT.js on the web.
///
/// Compilation and inference are Promise-based in LiteRT.js, so only the
/// asynchronous half of the CompiledModel API is available here:
/// [fromBufferAsync] / [fromBufferWithGpuFallbackAsync] to build a model and
/// [runAsync] to run it. The synchronous [fromFile], [fromBuffer],
/// [fromBufferWithGpuFallback], and [run] throw [UnsupportedError] on the web;
/// portable code should use the async variants, which work on every platform.
///
/// [Accelerator.cpu] maps to the LiteRT.js WASM backend and [Accelerator.gpu]
/// to WebGPU. When a requested accelerator set contains both, WebGPU is tried
/// first with a WASM fallback, mirroring the native GPU-with-CPU-fallback
/// behavior; [accelerators] reports which backend actually compiled.
///
/// Failures at inference time (most commonly WebGPU device loss or GPU OOM)
/// surface as [LiteRtRuntimeError], matching the web `LiteRtInterpreter`, so
/// callers can dispose the model and rebuild it on the CPU backend.
class CompiledModel {
  CompiledModel._(
    this._model,
    this._inputs,
    this._outputs,
    this._accelerator,
    this._accelerators,
  );

  final lrt.CompiledModelJS _model;
  final List<_TensorDetails> _inputs;
  final List<_TensorDetails> _outputs;
  final Accelerator _accelerator;
  final Set<Accelerator> _accelerators;

  // Whole runAsync cycles are serialized in FIFO order to match the native
  // implementation's semantics (there they share native I/O buffers; here we
  // avoid overlapping run() calls on one LiteRT.js model).
  final AsyncLock _runLock = AsyncLock();
  bool _dispatchInFlight = false;
  bool _closed = false;

  String get _acceleratorName =>
      _accelerator == Accelerator.gpu ? 'webgpu' : 'wasm';

  /// Not supported on the web: there is no file system. Load the model bytes
  /// (e.g. via `rootBundle.load` or an HTTP fetch) and use [fromBufferAsync].
  static CompiledModel fromFile(
    String path, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    throw UnsupportedError(
      'CompiledModel.fromFile is not supported on the web (no file system). '
      'Load the model bytes and use CompiledModel.fromBufferAsync instead.',
    );
  }

  /// Not supported on the web: LiteRT.js compilation is Promise-based and
  /// cannot complete synchronously. Use [fromBufferAsync].
  static CompiledModel fromBuffer(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    throw UnsupportedError(
      'CompiledModel.fromBuffer is synchronous and cannot be implemented on '
      'the web, where LiteRT.js compilation is Promise-based. Use '
      'CompiledModel.fromBufferAsync instead.',
    );
  }

  /// Not supported on the web. Use [fromBufferWithGpuFallbackAsync].
  static CompiledModel fromBufferWithGpuFallback(
    Uint8List bytes, {
    bool forceCpu = false,
    Precision precision = Precision.fp32,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
    void Function(Object error)? onFallback,
  }) {
    throw UnsupportedError(
      'CompiledModel.fromBufferWithGpuFallback is synchronous and cannot be '
      'implemented on the web, where LiteRT.js compilation is Promise-based. '
      'Use CompiledModel.fromBufferWithGpuFallbackAsync instead.',
    );
  }

  /// Creates a compiled model from model bytes via LiteRT.js.
  ///
  /// [accelerators] selects the backend: `{cpu}` compiles on WASM, `{gpu}` on
  /// WebGPU (throwing if that fails), and `{gpu, cpu}` tries WebGPU first and
  /// falls back to WASM. [Accelerator.npu] is not available on the web.
  ///
  /// [precision] is accepted for API parity but ignored: LiteRT.js does not
  /// expose a precision option, so WebGPU kernels run at their default
  /// precision. [tensorBufferMode] must be [TensorBufferMode.managed];
  /// host-memory buffers are a native-only concept.
  ///
  /// When the set contains both accelerators, the WebGPU attempt is also
  /// bounded by a watchdog: if the LiteRT.js compile promise neither
  /// resolves nor rejects within 60 seconds (observed on GPU-less headless
  /// Chrome, where the promise can occasionally never settle), the attempt
  /// is abandoned and compilation proceeds on WASM. Strict `{gpu}` requests
  /// are never timed out; they surface whatever the runtime does.
  ///
  /// The first call auto-loads the LiteRT.js runtime (see
  /// `configureLiteRtWebLoader` to self-host or disable that).
  static Future<CompiledModel> fromBufferAsync(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) => _fromBufferAsyncImpl(
    bytes,
    accelerators: accelerators,
    tensorBufferMode: tensorBufferMode,
  );

  static Future<CompiledModel> _fromBufferAsyncImpl(
    Uint8List bytes, {
    required Set<Accelerator> accelerators,
    required TensorBufferMode tensorBufferMode,
    void Function(Object error)? onGpuFallback,
  }) async {
    if (accelerators.isEmpty) {
      throw ArgumentError.value(
        accelerators,
        'accelerators',
        'Must contain at least one accelerator.',
      );
    }
    if (accelerators.contains(Accelerator.npu)) {
      throw ArgumentError.value(
        accelerators,
        'accelerators',
        'Accelerator.npu is not supported on the web.',
      );
    }
    if (tensorBufferMode != TensorBufferMode.managed) {
      throw UnsupportedError(
        'TensorBufferMode.${tensorBufferMode.name} is not supported on the '
        'web; use TensorBufferMode.managed.',
      );
    }
    if (bytes.isEmpty) {
      throw ArgumentError.value(
        bytes.length,
        'bytes.length',
        'Must be non-zero.',
      );
    }

    await lrt.waitForLiteRt();

    lrt.CompiledModelJS compiled;
    Accelerator requested;
    if (accelerators.contains(Accelerator.gpu)) {
      final bool canFallBack = accelerators.contains(Accelerator.cpu);
      try {
        // Bound the WebGPU attempt only when a WASM fallback exists: the
        // watchdog turns a wedged compile into a fallback, never an error.
        compiled = canFallBack
            ? await _compileGpuWithWatchdog(bytes)
            : await _compile(bytes, 'webgpu');
        requested = Accelerator.gpu;
      } catch (e) {
        if (!canFallBack) rethrow;
        onGpuFallback?.call(e);
        compiled = await _compile(bytes, 'wasm');
        requested = Accelerator.cpu;
      }
    } else {
      compiled = await _compile(bytes, 'wasm');
      requested = Accelerator.cpu;
    }
    return _fromCompiled(compiled, requested);
  }

  /// How long a WebGPU compile attempt may stay unsettled before a pending
  /// WASM fallback abandons it. LiteRT.js 2.4.0's compile promise can, very
  /// rarely, neither resolve nor reject on machines without a usable GPU
  /// (observed on GPU-less headless Chrome in CI); without a bound, the
  /// fallback contract of [fromBufferWithGpuFallbackAsync] and of a
  /// `{gpu, cpu}` accelerator set would silently hang with it.
  @visibleForTesting
  static Duration debugGpuCompileWatchdog = const Duration(seconds: 60);

  /// Compiles on WebGPU, throwing [TimeoutException] if the LiteRT.js
  /// compile promise does not settle within [debugGpuCompileWatchdog].
  /// Callers only use this when they hold a WASM fallback. The abandoned
  /// promise cannot be cancelled; if it settles later, its model is deleted
  /// so a late success cannot leak a WebGPU model.
  static Future<lrt.CompiledModelJS> _compileGpuWithWatchdog(Uint8List bytes) {
    final attempt = _compile(bytes, 'webgpu');
    return attempt.timeout(
      debugGpuCompileWatchdog,
      onTimeout: () {
        attempt
            .then((m) => m.delete())
            .catchError((Object _) {}); // Late failures need no cleanup.
        throw TimeoutException(
          'LiteRT.js loadAndCompile on webgpu did not settle within '
          '${debugGpuCompileWatchdog.inSeconds}s; treating WebGPU as '
          'unavailable.',
        );
      },
    );
  }

  /// Compiles via LiteRT.js, normalizing failures to [StateError] so web
  /// compile errors carry the same type as native ones.
  static Future<lrt.CompiledModelJS> _compile(
    Uint8List bytes,
    String accelerator,
  ) async {
    try {
      return await lrt.loadAndCompile(bytes, accelerator: accelerator);
    } on StateError {
      rethrow;
    } catch (e) {
      throw StateError('LiteRT.js loadAndCompile failed on $accelerator: $e');
    }
  }

  /// Creates a compiled model from [bytes], preferring WebGPU with a WASM
  /// fallback.
  ///
  /// Web counterpart of the native `fromBufferWithGpuFallback`: tries the
  /// WebGPU backend and retries on WASM if compilation throws (no WebGPU
  /// support, unsupported op, adapter failure) or fails to settle within
  /// the 60-second compile watchdog, so callers always get a working model.
  /// Pass [forceCpu] to skip the WebGPU attempt. [onFallback] is invoked
  /// with the WebGPU error ([TimeoutException] for a watchdog abandonment)
  /// when the WASM retry happens.
  static Future<CompiledModel> fromBufferWithGpuFallbackAsync(
    Uint8List bytes, {
    bool forceCpu = false,
    Precision precision = Precision.fp32,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
    void Function(Object error)? onFallback,
  }) {
    if (forceCpu) {
      return fromBufferAsync(
        bytes,
        accelerators: const {Accelerator.cpu},
        tensorBufferMode: tensorBufferMode,
      );
    }
    return _fromBufferAsyncImpl(
      bytes,
      accelerators: const {Accelerator.gpu, Accelerator.cpu},
      tensorBufferMode: tensorBufferMode,
      onGpuFallback: onFallback,
    );
  }

  static CompiledModel _fromCompiled(
    lrt.CompiledModelJS compiled,
    Accelerator requested,
  ) {
    try {
      final inputs = _parseDetails(compiled.getInputDetails(), 'input');
      final outputs = _parseDetails(compiled.getOutputDetails(), 'output');
      final resolved = _resolvedAccelerator(compiled, requested);
      final acceleratorSet = <Accelerator>{resolved};
      if (resolved == Accelerator.gpu && !_isFullyAccelerated(compiled)) {
        // The runtime placed some ops on the CPU (partial delegation),
        // mirroring how a native {gpu, cpu} set reports op-level fallback.
        acceleratorSet.add(Accelerator.cpu);
      }
      return CompiledModel._(
        compiled,
        inputs,
        outputs,
        resolved,
        Set.unmodifiable(acceleratorSet),
      );
    } catch (_) {
      compiled.delete();
      rethrow;
    }
  }

  /// The accelerator the model was actually compiled with.
  ///
  /// LiteRT.js 2.4.0 throws when a WebGPU request cannot compile, but newer
  /// builds (reachable via `configureLiteRtWebLoader`) can silently recompile
  /// on WASM instead. The final compile options live on the model, so prefer
  /// those over [requested]; fall back to [requested] if the property shape
  /// ever changes.
  static Accelerator _resolvedAccelerator(
    lrt.CompiledModelJS compiled,
    Accelerator requested,
  ) {
    try {
      final options = compiled.getProperty<JSObject?>('options'.toJS);
      final accelerator = options?.getProperty<JSAny?>('accelerator'.toJS);
      if (accelerator != null && accelerator.isA<JSString>()) {
        switch ((accelerator as JSString).toDart) {
          case 'wasm':
            return Accelerator.cpu;
          case 'webgpu':
            return Accelerator.gpu;
        }
      }
    } catch (_) {
      // Fall through to the requested accelerator.
    }
    return requested;
  }

  /// Whether every op was placed on the requested accelerator. Unknown (older
  /// or reshaped runtimes) is treated as fully accelerated, matching LiteRT.js
  /// 2.4.0 semantics where a WebGPU compile that cannot place ops throws.
  static bool _isFullyAccelerated(lrt.CompiledModelJS compiled) {
    try {
      final value = compiled.getProperty<JSAny?>('isFullyAccelerated'.toJS);
      if (value != null && value.isA<JSBoolean>()) {
        return (value as JSBoolean).toDart;
      }
    } catch (_) {
      // Treat as fully accelerated.
    }
    return true;
  }

  static List<_TensorDetails> _parseDetails(
    JSArray<JSObject> detailsJs,
    String kind,
  ) {
    final details = detailsJs.toDart;
    return List<_TensorDetails>.generate(details.length, (i) {
      final d = _TensorDetails.parse(details[i]);
      if (d.dtype != 'float32') {
        throw UnsupportedError(
          'CompiledModel.runAsync supports Float32 tensors only; $kind[$i] '
          'has dtype ${d.dtype}.',
        );
      }
      if (d.shape.any((dim) => dim <= 0)) {
        throw UnsupportedError(
          'CompiledModel on the web requires fully static tensor shapes; '
          '$kind[$i] has shape ${d.shape}.',
        );
      }
      return d;
    }, growable: false);
  }

  /// Tensor buffer allocation mode used by this model (always
  /// [TensorBufferMode.managed] on the web).
  TensorBufferMode get tensorBufferMode => TensorBufferMode.managed;

  /// Accelerators this model was compiled with.
  ///
  /// On the web this reports what LiteRT.js actually resolved rather than the
  /// requested set: `{Accelerator.gpu}` for a fully accelerated WebGPU model,
  /// `{Accelerator.cpu}` for WASM, and `{Accelerator.gpu, Accelerator.cpu}`
  /// when the runtime reports a WebGPU model as only partially accelerated
  /// (some ops delegated to WASM). It may differ from the requested set when
  /// WebGPU compilation failed and the WASM fallback was used.
  Set<Accelerator> get accelerators => _accelerators;

  /// Whether the **whole** graph runs on the selected accelerator.
  ///
  /// False is ambiguous: it covers both partial delegation and no acceleration
  /// at all, so this is not a way to tell whether acceleration happened. See
  /// the native implementation for the reasoning and the alternative.
  ///
  /// Reports true when the runtime does not expose the property, matching how
  /// [accelerators] only adds a CPU entry when partial delegation is
  /// affirmatively reported.
  bool get isFullyAccelerated => _isFullyAccelerated(_model);

  /// Number of input tensors for the default signature.
  int get inputCount => _inputs.length;

  /// Number of output tensors for the default signature.
  int get outputCount => _outputs.length;

  /// Byte size of each input tensor, index-aligned with [runAsync].
  List<int> get inputByteSizes => List.unmodifiable(
    _inputs.map((d) => d.floatCount * Float32List.bytesPerElement),
  );

  /// Byte size of each output tensor, index-aligned with [runAsync]'s result.
  List<int> get outputByteSizes => List.unmodifiable(
    _outputs.map((d) => d.floatCount * Float32List.bytesPerElement),
  );

  /// Not supported on the web: LiteRT.js inference is Promise-based and
  /// cannot complete synchronously. Use [runAsync].
  List<Float32List> run(List<Float32List> inputs) {
    throw UnsupportedError(
      'CompiledModel.run is synchronous and cannot be implemented on the web, '
      'where LiteRT.js inference is Promise-based. Use runAsync instead.',
    );
  }

  /// Runs inference with Float32 input tensors and returns Float32 outputs.
  ///
  /// Concurrent calls are serialized in FIFO order, matching the native
  /// implementation. [inputs] are copied into LiteRT.js tensors when this
  /// call's turn in the queue arrives, not at call time; do not mutate them
  /// until the returned future completes.
  ///
  /// On the WASM backend the returned future is asynchronous in shape only:
  /// the non-threaded WASM build still computes on the main thread. WebGPU
  /// executes genuinely off the main thread.
  Future<List<Float32List>> runAsync(List<Float32List> inputs) async {
    _ensureOpen();
    if (inputs.length != inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected $inputCount input tensors.',
      );
    }

    return _runLock.run(() async {
      _ensureOpen();
      _dispatchInFlight = true;
      try {
        return await _runOnce(inputs);
      } finally {
        _dispatchInFlight = false;
      }
    });
  }

  Future<List<Float32List>> _runOnce(List<Float32List> inputs) async {
    final tensorsJs = <lrt.LiteRtTensorJS>[];
    JSArray<lrt.LiteRtTensorJS> resultJs;
    try {
      for (var i = 0; i < inputs.length; i++) {
        final expectedFloats = _inputs[i].floatCount;
        if (inputs[i].length != expectedFloats) {
          throw ArgumentError.value(
            inputs[i].length,
            'inputs[$i].length',
            'Expected $expectedFloats float32 values.',
          );
        }
        tensorsJs.add(lrt.makeInputTensor(inputs[i], _inputs[i].shape));
      }
      final JSAny callArg = tensorsJs.length == 1
          ? tensorsJs[0] as JSAny
          : tensorsJs.cast<JSAny>().toJS;
      try {
        resultJs = await _model.run(callArg).toDart;
      } catch (e) {
        // WebGPU can fail at inference time (device lost, GPU OOM) rather
        // than at compile time; surface the same typed error as the web
        // LiteRtInterpreter so callers can rebuild on the CPU backend.
        throw LiteRtRuntimeError(accelerator: _acceleratorName, cause: e);
      }
    } finally {
      for (final t in tensorsJs) {
        t.delete();
      }
    }

    final resultDart = resultJs.toDart;
    try {
      final outputs = <Float32List>[];
      for (final t in resultDart) {
        try {
          outputs.add(await _readOwnedFloat32(t));
        } catch (e) {
          if (e is LiteRtRuntimeError) rethrow;
          throw LiteRtRuntimeError(accelerator: _acceleratorName, cause: e);
        }
      }
      return outputs;
    } finally {
      for (final t in resultDart) {
        t.delete();
      }
    }
  }

  /// Reads [t]'s data as a caller-owned [Float32List]. The sync
  /// `toTypedArray()` path returns a view into WASM-heap memory that dies
  /// with the tensor, so it is copied; the async `data()` path already
  /// returns a fresh copy.
  static Future<Float32List> _readOwnedFloat32(lrt.LiteRtTensorJS t) async {
    try {
      final JSAny v = t.toTypedArray();
      if (v.isA<JSFloat32Array>()) {
        return Float32List.fromList((v as JSFloat32Array).toDart);
      }
    } catch (_) {
      // GPU tensor: fall through to async data().
    }
    final JSAny v = await t.dataAsync().toDart;
    if (v.isA<JSFloat32Array>()) {
      return (v as JSFloat32Array).toDart;
    }
    throw StateError(
      'LiteRT.js Tensor.data() returned unexpected type ${v.runtimeType}',
    );
  }

  /// Host-memory tensor buffers are native-only; always throws on the web.
  void writeInput(int index, void Function(Float32List input) write) {
    _hostMemoryUnsupported('writeInput');
  }

  /// Host-memory tensor buffers are native-only; always throws on the web.
  void dispatch() {
    _hostMemoryUnsupported('dispatch');
  }

  /// Host-memory tensor buffers are native-only; always throws on the web.
  Future<void> dispatchAsync() async {
    _hostMemoryUnsupported('dispatchAsync');
  }

  /// Host-memory tensor buffers are native-only; always throws on the web.
  R readOutput<R>(int index, R Function(Float32List output) read) {
    _hostMemoryUnsupported('readOutput');
  }

  /// Releases the underlying LiteRT.js model.
  ///
  /// Throws [StateError] while a [runAsync] call is in flight; await pending
  /// futures first.
  void close() {
    if (_closed) return;
    if (_dispatchInFlight) {
      throw StateError(
        'CompiledModel.close() called while an async dispatch is in flight.',
      );
    }
    _closed = true;
    _model.delete();
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('CompiledModel is already closed.');
    }
  }

  Never _hostMemoryUnsupported(String method) {
    _ensureOpen();
    throw StateError(
      'CompiledModel.$method requires '
      'TensorBufferMode.${TensorBufferMode.hostMemory.name}, which is not '
      'supported on the web.',
    );
  }
}

class _TensorDetails {
  _TensorDetails({required this.name, required this.dtype, required this.shape})
    : floatCount = shape.fold(1, (a, b) => a * b);

  final String name;
  final String dtype;
  final List<int> shape;
  final int floatCount;

  static _TensorDetails parse(JSObject d) {
    final JSAny? shapeAny = d.getProperty('shape'.toJS);
    final List<int> shape;
    if (shapeAny != null && shapeAny.isA<JSInt32Array>()) {
      shape = (shapeAny as JSInt32Array).toDart.toList(growable: false);
    } else {
      final dyn = shapeAny?.dartify();
      shape = (dyn as List).map((e) => (e as num).toInt()).toList();
    }
    final String dtype =
        ((d.getProperty('dtype'.toJS) as JSString?) ?? 'float32'.toJS).toDart;
    final String name = (d.getProperty('name'.toJS) as JSString?)?.toDart ?? '';
    return _TensorDetails(name: name, dtype: dtype, shape: shape);
  }
}
