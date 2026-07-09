// LiteRT.js-backed alternative to [Interpreter] for the web build.
//
// API surface chosen to match what `pose_detection`'s YOLOv8 detector needs:
//   - fromBytes(Uint8List)
//   - allocateTensors() (no-op)
//   - getInputTensor(0).shape
//   - getOutputTensors() / getOutputTensor(int)
//   - runForMultipleInputs(List<Object>, Map<int, Object>)
//
// The native flutter_litert Interpreter uses tflite-js under the hood;
// this one uses Google's official LiteRT.js. Both are runtime-selectable
// at the call site.

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;

import 'tensor.dart';
import 'js_interop/litertjs_bindings.dart' as lrt;

/// Thrown by [LiteRtInterpreter.runForMultipleInputs] when the underlying
/// LiteRT.js call fails at inference time (as opposed to at compile time).
///
/// Wraps the original JS-side error in [cause] and reports the
/// [accelerator] that was active when the failure occurred. Consumers
/// running on `'webgpu'` typically respond by disposing all related
/// interpreters and re-initializing them with `accelerator: 'wasm'`,
/// which is robust to GPU OOM, device loss, and other GPU-specific
/// failure modes that can fire mid-stream.
class LiteRtRuntimeError extends Error {
  /// The accelerator that was active when the error fired
  /// (`'webgpu'` or `'wasm'`).
  final String accelerator;

  /// The original JS-side error, exposed as a Dart `Object` so callers
  /// can `toString()` it for logging without needing JS interop.
  final Object cause;

  /// Human-readable message synthesized from [cause].
  final String message;

  LiteRtRuntimeError({
    required this.accelerator,
    required this.cause,
    String? message,
  }) : message = message ?? cause.toString();

  @override
  String toString() =>
      'LiteRtRuntimeError(accelerator: $accelerator, cause: $message)';
}

/// LiteRT.js-backed interpreter. Construct via [fromBytes].
class LiteRtInterpreter {
  final lrt.CompiledModelJS _model;
  final List<_TensorMeta> _inputs;
  final List<Tensor> _outTensors;
  final String _activeAccelerator;
  bool _disposed = false;

  int _lastInferenceDurationMicroseconds = 0;

  /// Microseconds spent inside the most recent `run()` call (LiteRT.js side).
  int get lastInferenceDurationMicroseconds =>
      _lastInferenceDurationMicroseconds;

  /// Microseconds spent inside the most recent `run()` call (LiteRT.js side).
  ///
  /// Deprecated in favor of [lastInferenceDurationMicroseconds], which uses a
  /// platform-neutral name and consistent microseconds casing.
  @Deprecated(
    'Use lastInferenceDurationMicroseconds instead. '
    'This alias will be removed in a future major release.',
  )
  int get lastNativeInferenceDurationMicroSeconds =>
      lastInferenceDurationMicroseconds;

  @Deprecated(
    'Use lastInferenceDurationMicroseconds instead. '
    'This alias will be removed in a future major release.',
  )
  set lastNativeInferenceDurationMicroSeconds(int value) {
    _lastInferenceDurationMicroseconds = value;
  }

  LiteRtInterpreter._(
    this._model, {
    required List<_TensorMeta> inputs,
    required List<_TensorMeta> outputs,
    required String activeAccelerator,
  }) : _inputs = inputs,
       _activeAccelerator = activeAccelerator,
       _outTensors = List<Tensor>.unmodifiable(
         outputs.map(
           (m) =>
               Tensor.fromMetadata(name: m.name, type: m.type, shape: m.shape),
         ),
       );

  /// The accelerator that was actually used to compile this interpreter.
  ///
  /// This is `'webgpu'` or `'wasm'`. May differ from the [accelerator]
  /// argument passed to [fromBytes] if the requested backend's compile step
  /// failed and we fell back. Consumers should display this in UI / logs so
  /// users know which path their inference is taking.
  String get activeAccelerator => _activeAccelerator;

  /// Compiles a .tflite model from raw bytes via LiteRT.js.
  ///
  /// [accelerator] is `'webgpu'` or `'wasm'`; it defaults to `'wasm'`.
  /// Pass `'webgpu'` when you want GPU acceleration.
  /// Falls back from webgpu to wasm if webgpu compile fails. Inspect
  /// [activeAccelerator] on the returned interpreter to see which one
  /// was actually used.
  static Future<LiteRtInterpreter> fromBytes(
    Uint8List bytes, {
    String accelerator = 'wasm',
  }) async {
    await lrt.waitForLiteRt();

    lrt.CompiledModelJS compiled;
    String resolved = accelerator;
    try {
      compiled = await lrt.loadAndCompile(bytes, accelerator: accelerator);
    } catch (e) {
      if (accelerator == 'webgpu') {
        // Fall back: not all ops are supported on webgpu. WASM should
        // accept everything that tflite-js accepts.
        debugPrint(
          'flutter_litert: WebGPU compile failed, falling back to WASM '
          '(inspect activeAccelerator on the returned interpreter). '
          'Cause: $e',
        );
        compiled = await lrt.loadAndCompile(bytes, accelerator: 'wasm');
        resolved = 'wasm';
      } else {
        rethrow;
      }
    }

    final inputs = compiled
        .getInputDetails()
        .toDart
        .map(_TensorMeta.fromDetails)
        .toList(growable: false);
    final outputs = compiled
        .getOutputDetails()
        .toDart
        .map(_TensorMeta.fromDetails)
        .toList(growable: false);

    return LiteRtInterpreter._(
      compiled,
      inputs: inputs,
      outputs: outputs,
      activeAccelerator: resolved,
    );
  }

  bool get isClosed => _disposed;

  void close() {
    if (_disposed) return;
    _disposed = true;
    _model.delete();
  }

  /// No-op (LiteRT.js manages tensor allocation internally).
  void allocateTensors() {}

  Tensor getInputTensor(int index) {
    final m = _inputs[index];
    return Tensor.fromMetadata(name: m.name, type: m.type, shape: m.shape);
  }

  List<Tensor> getOutputTensors() => _outTensors;

  Tensor getOutputTensor(int index) => _outTensors[index];

  /// Runs inference and copies float32 outputs into [outputs] (matching
  /// `flutter_litert`'s `Interpreter.runForMultipleInputs` semantics).
  ///
  /// [inputs] entries may be `Float32List`, `ByteBuffer`, or `Uint8List` whose
  /// underlying bytes are interpreted as float32.
  ///
  /// [outputs] is a map from output index to a destination buffer. Supports
  /// `Float32List` (preferred, fast bulk copy), `ByteBuffer`, and the legacy
  /// nested `List<List<List<double>>>` shape used by tflite-js callers.
  Future<void> runForMultipleInputs(
    List<Object> inputs,
    Map<int, Object> outputs,
  ) async {
    if (_disposed) {
      throw StateError('LiteRtInterpreter has been closed.');
    }
    if (inputs.length != _inputs.length) {
      throw ArgumentError(
        'Expected ${_inputs.length} inputs, got ${inputs.length}.',
      );
    }

    final List<lrt.LiteRtTensorJS> tensorsJs = <lrt.LiteRtTensorJS>[];
    for (int i = 0; i < inputs.length; i++) {
      final Float32List asFloat = _asFloat32(inputs[i]);
      tensorsJs.add(lrt.makeInputTensor(asFloat, _inputs[i].shape));
    }

    final start = DateTime.now().microsecondsSinceEpoch;
    JSArray<lrt.LiteRtTensorJS>? resultJs;
    try {
      try {
        final JSAny callArg = tensorsJs.length == 1
            ? tensorsJs[0] as JSAny
            : tensorsJs.cast<JSAny>().toJS;
        resultJs = await _model.run(callArg).toDart;
      } catch (e) {
        // Inference-time failure (vs. compile-time, which is handled in
        // [fromBytes]). Most commonly fires on the WebGPU path with errors
        // like `GPUOutOfMemoryError`, `GPUValidationError`, or device-lost.
        // Surface a typed exception so consumers can decide whether to
        // dispose and re-init on the WASM path.
        throw LiteRtRuntimeError(accelerator: _activeAccelerator, cause: e);
      }
    } finally {
      for (final t in tensorsJs) {
        t.delete();
      }
    }
    _lastInferenceDurationMicroseconds =
        DateTime.now().microsecondsSinceEpoch - start;

    final List<lrt.LiteRtTensorJS> resultDart = resultJs.toDart;
    try {
      for (int i = 0; i < resultDart.length; i++) {
        if (!outputs.containsKey(i)) continue;
        final Float32List flat;
        try {
          flat = await _readFloat32(resultDart[i]);
        } catch (e) {
          // Readback-time failure; also classify as runtime so consumers
          // can swap backends on it.
          throw LiteRtRuntimeError(accelerator: _activeAccelerator, cause: e);
        }
        _writeOutput(flat, outputs[i]!);
      }
    } finally {
      for (final t in resultDart) {
        t.delete();
      }
    }
  }

  /// Reads a tensor's data as Float32List, working for both CPU and GPU
  /// memory backings. Tries the sync path first.
  static Future<Float32List> _readFloat32(lrt.LiteRtTensorJS t) async {
    try {
      final JSAny v = t.toTypedArray();
      if (v.isA<JSFloat32Array>()) {
        return (v as JSFloat32Array).toDart;
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

  // --- helpers ---

  static Float32List _asFloat32(Object input) {
    if (input is Float32List) return input;
    if (input is ByteBuffer) return input.asFloat32List();
    if (input is Uint8List) {
      return Float32List.view(
        input.buffer,
        input.offsetInBytes,
        input.lengthInBytes >> 2,
      );
    }
    throw ArgumentError(
      'Unsupported input type: ${input.runtimeType}. '
      'LiteRtInterpreter expects Float32List / ByteBuffer / Uint8List.',
    );
  }

  /// Writes `flat` into `dst`, supporting both flat `Float32List` and the
  /// nested `List<List<List<double>>>` shape used by older callers.
  static void _writeOutput(Float32List flat, Object dst) {
    if (dst is Float32List) {
      dst.setRange(0, flat.length, flat);
      return;
    }
    if (dst is ByteBuffer) {
      final view = dst.asFloat32List();
      view.setRange(0, flat.length, flat);
      return;
    }
    if (dst is List) {
      _fillNested(dst, flat, 0);
      return;
    }
    throw ArgumentError('Unsupported output buffer type: ${dst.runtimeType}.');
  }

  static int _fillNested(List dst, Float32List flat, int offset) {
    if (dst.isEmpty) return offset;
    final first = dst.first;
    if (first is List) {
      for (int i = 0; i < dst.length; i++) {
        offset = _fillNested(dst[i] as List, flat, offset);
      }
    } else {
      // dst is List<double>
      final List<double> leaf = dst.cast<double>();
      for (int i = 0; i < leaf.length; i++) {
        leaf[i] = flat[offset + i];
      }
      offset += leaf.length;
    }
    return offset;
  }
}

class _TensorMeta {
  final String name;
  final TensorType type;
  final List<int> shape;
  _TensorMeta({required this.name, required this.type, required this.shape});

  static _TensorMeta fromDetails(JSObject d) {
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
    final TensorType type;
    switch (dtype) {
      case 'float32':
        type = TensorType.float32;
        break;
      case 'int32':
        type = TensorType.int32;
        break;
      default:
        type = TensorType.float32;
    }
    final String name = (d.getProperty('name'.toJS) as JSString?)?.toDart ?? '';
    return _TensorMeta(name: name, type: type, shape: shape);
  }
}
