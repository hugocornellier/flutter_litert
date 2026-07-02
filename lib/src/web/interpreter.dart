import 'dart:typed_data';

import 'interpreter_options.dart';
import 'js_interop/model_tensor_info.dart';
import 'js_interop/tflite_js.dart';
import 'js_interop/tfjs_tensor.dart';
import 'model.dart';
import 'signature_runner.dart';
import 'tensor.dart';
import 'version.dart' as tflite_version;
import '../util/byte_conversion_utils_web.dart';
import '../util/flutter_asset_utils.dart';

/// Web implementation of Interpreter.
///
/// Uses the TFLite.js WASM runtime for inference. The supported subset of the
/// API (`fromAsset`, `fromBytes`, `run`, `runForMultipleInputs`, and the tensor
/// getters) matches the native Interpreter, so common inference code needs no
/// conditional branching. Native-only members (`fromFile`, `fromBuffer`,
/// `fromAddress`, `resizeInputTensor`, the signature-runner and variable-tensor
/// APIs, and `address`) throw [UnsupportedError] on web.
class Interpreter {
  TFLiteModel _model;
  bool _deleted = false;
  int _lastInferenceDurationMicroseconds = 0;

  /// Returns the TFLite web runtime version string.
  static String get version => tflite_version.version;

  List<Tensor>? _inputTensors;
  List<Tensor>? _outputTensors;

  /// Duration of the last inference call in microseconds.
  int get lastInferenceDurationMicroseconds =>
      _lastInferenceDurationMicroseconds;

  /// Duration of the last inference call in microseconds.
  ///
  /// Deprecated in favor of [lastInferenceDurationMicroseconds], which uses a
  /// platform-neutral name and consistent microseconds casing.
  @Deprecated(
    'Use lastInferenceDurationMicroseconds instead. '
    'This alias will be removed in a future major release.',
  )
  int get lastNativeInferenceDurationMicroSeconds =>
      lastInferenceDurationMicroseconds;

  Interpreter._(this._model);

  /// Not supported on web. Use [fromAsset] or [fromBytes] instead.
  factory Interpreter.fromFile(
    dynamic modelFile, {
    InterpreterOptions? options,
  }) => throw UnsupportedError(
    'Interpreter.fromFile is not supported on web. '
    'Use Interpreter.fromAsset or Interpreter.fromBytes instead.',
  );

  /// Not supported on web.
  ///
  /// `fromBuffer` is a synchronous factory, but web model loading is
  /// asynchronous, so it cannot construct an interpreter. Throws
  /// [UnsupportedError]; use [fromAsset] or [fromBytes] (both async) instead.
  factory Interpreter.fromBuffer(
    Uint8List buffer, {
    InterpreterOptions? options,
  }) {
    throw UnsupportedError(
      'Interpreter.fromBuffer is synchronous but web loading is async. '
      'Use Interpreter.fromAsset or Interpreter.fromBytes instead.',
    );
  }

  /// Creates interpreter from an asset name.
  ///
  /// This is the primary way to create an interpreter on web.
  static Future<Interpreter> fromAsset(
    String assetName, {
    InterpreterOptions? options,
  }) async {
    final buffer = await loadAssetBytes(assetName);
    final model = await Model.fromBuffer(buffer);
    return Interpreter._(model.base);
  }

  /// Creates interpreter from raw bytes (async web-compatible version).
  static Future<Interpreter> fromBytes(
    Uint8List buffer, {
    InterpreterOptions? options,
  }) async {
    final model = await Model.fromBuffer(buffer);
    return Interpreter._(model.base);
  }

  /// Not supported on web (no pointer addresses).
  factory Interpreter.fromAddress(
    int address, {
    bool allocated = true,
    bool deleted = false,
  }) => throw UnsupportedError(
    'Interpreter.fromAddress is not supported on web.',
  );

  /// Closes the interpreter.
  void close() {
    _deleted = true;
    _inputTensors = null;
    _outputTensors = null;
  }

  /// No-op on web (TFLite.js handles allocation internally).
  void allocateTensors() {}

  /// No-op on web (inference is done through run/runForMultipleInputs).
  void invoke() {}

  /// Run inference for single input and output.
  void run(Object input, Object output) {
    var map = <int, Object>{};
    map[0] = output;
    runForMultipleInputs([input], map);
  }

  /// Run inference for multiple inputs and outputs.
  void runForMultipleInputs(List<Object> inputs, Map<int, Object> outputs) {
    if (outputs.isEmpty) {
      throw ArgumentError('Outputs should not be empty.');
    }
    runInference(inputs);
    var outputTensors = getOutputTensors();
    for (var i = 0; i < outputTensors.length; i++) {
      if (outputs.containsKey(i)) {
        outputTensors[i].copyTo(outputs[i]!);
      }
    }
  }

  /// Runs inference by converting inputs to JS tensors and calling predict.
  void runInference(List<Object> inputs) {
    if (inputs.isEmpty) {
      throw ArgumentError('Inputs should not be empty.');
    }

    final modelInputs = _model.inputs;

    // Convert each input to a JSTensor
    final List<JSTensor> jsTensors = [];
    for (int i = 0; i < inputs.length; i++) {
      final input = inputs[i];
      final info = modelInputs[i];
      final shape = info.shape ?? [];

      Uint8List bytes;
      if (input is Uint8List) {
        bytes = input;
      } else if (input is ByteBuffer) {
        bytes = input.asUint8List();
      } else {
        final tensorType = _tensorTypeFromInfo(info);
        bytes = ByteConversionUtils.convertObjectToBytes(input, tensorType);
      }

      // Create JS tensor from typed data
      final jsTensor = _createJSTensorFromBytes(bytes, shape, info);
      jsTensors.add(jsTensor);
    }

    var inferenceStartNanos = DateTime.now().microsecondsSinceEpoch;

    // Run prediction
    final dynamic result;
    if (jsTensors.length == 1) {
      result = _model.predict<dynamic>(jsTensors[0]);
    } else {
      result = _model.predict<dynamic>(jsTensors);
    }

    _lastInferenceDurationMicroseconds =
        DateTime.now().microsecondsSinceEpoch - inferenceStartNanos;

    // Extract output data
    final modelOutputs = _model.outputs;
    _outputTensors = [];

    if (result is List) {
      for (int i = 0; i < result.length; i++) {
        final outputJsTensor = result[i] as JSTensor;
        final info = i < modelOutputs.length ? modelOutputs[i] : null;
        _outputTensors!.add(_tensorFromJSTensor(outputJsTensor, info));
        outputJsTensor.dispose();
      }
    } else {
      // ignore: invalid_runtime_check_with_js_interop_types
      final jsTensor = result as JSTensor;
      final info = modelOutputs.isNotEmpty ? modelOutputs[0] : null;
      _outputTensors!.add(_tensorFromJSTensor(jsTensor, info));
      jsTensor.dispose();
    }

    // Dispose input tensors
    for (final t in jsTensors) {
      t.dispose();
    }
  }

  /// Gets input tensors (metadata-only on web).
  List<Tensor> getInputTensors() {
    if (_inputTensors != null) return _inputTensors!;

    final modelInputs = _model.inputs;
    _inputTensors = List.generate(modelInputs.length, (i) {
      final info = modelInputs[i];
      return Tensor.fromMetadata(
        name: info.name,
        type: _tensorTypeFromInfo(info),
        shape: info.shape ?? [],
      );
    }, growable: false);

    return _inputTensors!;
  }

  /// Gets output tensors.
  List<Tensor> getOutputTensors() {
    if (_outputTensors != null) return _outputTensors!;

    // Return metadata-only tensors if inference hasn't run yet
    final modelOutputs = _model.outputs;
    return List.generate(modelOutputs.length, (i) {
      final info = modelOutputs[i];
      return Tensor.fromMetadata(
        name: info.name,
        type: _tensorTypeFromInfo(info),
        shape: info.shape ?? [],
      );
    }, growable: false);
  }

  /// Not supported on web, TFLite.js does not expose a tensor resize API.
  void resizeInputTensor(int tensorIndex, List<int> shape) =>
      throw UnsupportedError(
        'Interpreter.resizeInputTensor is not supported on web. '
        'TFLite.js uses fixed-shape WASM models with no resize API.',
      );

  /// Gets the input Tensor for the provided input index.
  Tensor getInputTensor(int index) {
    final tensors = getInputTensors();
    if (index < 0 || index >= tensors.length) {
      throw ArgumentError('Invalid input Tensor index: $index');
    }
    return tensors[index];
  }

  /// Gets the output Tensor for the provided output index.
  Tensor getOutputTensor(int index) {
    final tensors = getOutputTensors();
    if (index < 0 || index >= tensors.length) {
      throw ArgumentError('Invalid output Tensor index: $index');
    }
    return tensors[index];
  }

  /// Gets index of an input given the op name.
  int getInputIndex(String opName) {
    final inputTensors = getInputTensors();
    for (var i = 0; i < inputTensors.length; i++) {
      if (inputTensors[i].name == opName) return i;
    }
    throw ArgumentError("'$opName' is not a valid input name.");
  }

  /// Gets index of an output given the op name.
  int getOutputIndex(String opName) {
    final outputTensors = getOutputTensors();
    for (var i = 0; i < outputTensors.length; i++) {
      if (outputTensors[i].name == opName) return i;
    }
    throw ArgumentError("'$opName' is not a valid output name.");
  }

  /// No-op on web.
  void resetVariableTensors() {}

  /// Returns 0 on web (no variable tensor support).
  int getVariableTensorCount() => 0;

  /// Not supported on web.
  Tensor getVariableTensor(int index) => throw UnsupportedError(
    'Interpreter.getVariableTensor is not supported on web.',
  );

  /// Not supported on web (no pointer addresses).
  int get address =>
      throw UnsupportedError('Interpreter.address is not supported on web.');

  /// Whether the interpreter is still usable (not yet closed).
  ///
  /// On web, tensor allocation is a no-op, so this reports open/closed state
  /// (`!deleted`) rather than native tensor allocation.
  bool get isAllocated => !_deleted;

  /// Whether the interpreter has been deleted.
  bool get isDeleted => _deleted;

  // Helpers

  TensorType _tensorTypeFromInfo(ModelTensorInfo info) {
    switch (info.dataType) {
      case TFLiteDataType.float32:
        return TensorType.float32;
      case TFLiteDataType.int32:
        return TensorType.int32;
      case TFLiteDataType.bool:
        return TensorType.boolean;
      case TFLiteDataType.string:
        return TensorType.string;
      default:
        return TensorType.float32;
    }
  }

  JSTensor _createJSTensorFromBytes(
    Uint8List bytes,
    List<int> shape,
    ModelTensorInfo info,
  ) {
    switch (info.dataType) {
      case TFLiteDataType.float32:
        final floatData = Float32List.view(bytes.buffer);
        return JSTensor(floatData, shape: shape, type: TFLiteDataType.float32);
      case TFLiteDataType.int32:
        final intData = Int32List.view(bytes.buffer);
        return JSTensor(intData, shape: shape, type: TFLiteDataType.int32);
      default:
        final floatData = Float32List.view(bytes.buffer);
        return JSTensor(floatData, shape: shape, type: TFLiteDataType.float32);
    }
  }

  Tensor _tensorFromJSTensor(JSTensor jsTensor, ModelTensorInfo? info) {
    final name = info?.name ?? '';
    final shape = info?.shape ?? [];
    final type = info != null ? _tensorTypeFromInfo(info) : TensorType.float32;
    final tensor = Tensor.fromMetadata(name: name, type: type, shape: shape);

    // Single-copy readback for float32 outputs: read the JS Float32Array via
    // `.toDart`, then copy into an owned Float32List (the JS-side buffer is
    // freed by the caller's JSTensor.dispose()). This avoids the previous
    // `dataSync().dartify()` path which materialized a List<double> first.
    if (type == TensorType.float32) {
      final Float32List jsBacked = jsTensor.dataSyncFloat32();
      final Float32List owned = Float32List(jsBacked.length)
        ..setAll(0, jsBacked);
      tensor.data = owned.buffer.asUint8List();
      return tensor;
    }

    // Non-float32 fallback: legacy path via dartify().
    final data = jsTensor.dataSync<List<double>>();
    final float32Data = Float32List.fromList(data.cast<double>());
    tensor.data = float32Data.buffer.asUint8List();
    return tensor;
  }

  // ---------------------------------------------------------------------------
  // Signature / SignatureRunner APIs (not supported on web)
  // ---------------------------------------------------------------------------

  /// Not supported on web.
  int get signatureCount => throw UnsupportedError(
    'Interpreter.signatureCount is not supported on web.',
  );

  /// Not supported on web.
  String getSignatureKey(int index) => throw UnsupportedError(
    'Interpreter.getSignatureKey is not supported on web.',
  );

  /// Not supported on web.
  List<String> get signatureKeys => throw UnsupportedError(
    'Interpreter.signatureKeys is not supported on web.',
  );

  /// Not supported on web.
  SignatureRunner getSignatureRunner(String signatureKey) =>
      throw UnsupportedError(
        'Interpreter.getSignatureRunner is not supported on web. '
        'On-device training requires a native platform.',
      );
}
