// Unsupported platform stub for Interpreter.
//
// This file is used when compiling for platforms where neither dart:io
// nor dart:js_interop is available.

import 'dart:typed_data';

import 'interpreter_options.dart';
import 'signature_runner.dart';
import 'tensor.dart';

/// LiteRT interpreter for running inference on a model.
class Interpreter {
  /// Returns the LiteRT runtime version string.
  static String get version => throw UnsupportedError(
    'Interpreter.version is not supported on this platform',
  );

  int get lastInferenceDurationMicroseconds => throw UnsupportedError(
    'Interpreter.lastInferenceDurationMicroseconds is not supported on this platform',
  );

  @Deprecated(
    'Use lastInferenceDurationMicroseconds instead. '
    'This alias will be removed in a future major release.',
  )
  int get lastNativeInferenceDurationMicroSeconds => throw UnsupportedError(
    'Interpreter.lastNativeInferenceDurationMicroSeconds is not supported on this platform',
  );

  /// Creates [Interpreter] from a model file
  factory Interpreter.fromFile(
    dynamic modelFile, {
    InterpreterOptions? options,
  }) => throw UnsupportedError(
    'Interpreter.fromFile is not supported on this platform',
  );

  /// Creates interpreter from a [buffer]
  factory Interpreter.fromBuffer(
    Uint8List buffer, {
    InterpreterOptions? options,
  }) => throw UnsupportedError(
    'Interpreter.fromBuffer is not supported on this platform',
  );

  /// Creates interpreter from raw model bytes.
  static Future<Interpreter> fromBytes(
    Uint8List bytes, {
    InterpreterOptions? options,
  }) => throw UnsupportedError(
    'Interpreter.fromBytes is not supported on this platform',
  );

  /// Creates interpreter from a [assetName]
  static Future<Interpreter> fromAsset(
    String assetName, {
    InterpreterOptions? options,
  }) => throw UnsupportedError(
    'Interpreter.fromAsset is not supported on this platform',
  );

  /// Creates interpreter from an address.
  factory Interpreter.fromAddress(
    int address, {
    bool allocated = true,
    bool deleted = false,
  }) => throw UnsupportedError(
    'Interpreter.fromAddress is not supported on this platform',
  );

  /// Destroys the interpreter instance.
  void close() => throw UnsupportedError(
    'Interpreter.close is not supported on this platform',
  );

  /// Updates allocations for all tensors.
  void allocateTensors() => throw UnsupportedError(
    'Interpreter.allocateTensors is not supported on this platform',
  );

  /// Runs inference for the loaded graph.
  void invoke() => throw UnsupportedError(
    'Interpreter.invoke is not supported on this platform',
  );

  /// Run for single input and output
  void run(Object input, Object output) => throw UnsupportedError(
    'Interpreter.run is not supported on this platform',
  );

  /// Run for multiple inputs and outputs
  void runForMultipleInputs(List<Object> inputs, Map<int, Object> outputs) =>
      throw UnsupportedError(
        'Interpreter.runForMultipleInputs is not supported on this platform',
      );

  /// Just run inference
  void runInference(List<Object> inputs) => throw UnsupportedError(
    'Interpreter.runInference is not supported on this platform',
  );

  /// Gets all input tensors associated with the model.
  List<Tensor> getInputTensors() => throw UnsupportedError(
    'Interpreter.getInputTensors is not supported on this platform',
  );

  /// Gets all output tensors associated with the model.
  List<Tensor> getOutputTensors() => throw UnsupportedError(
    'Interpreter.getOutputTensors is not supported on this platform',
  );

  /// Resize input tensor for the given tensor index.
  void resizeInputTensor(int tensorIndex, List<int> shape) =>
      throw UnsupportedError(
        'Interpreter.resizeInputTensor is not supported on this platform',
      );

  /// Gets the input Tensor for the provided input index.
  Tensor getInputTensor(int index) => throw UnsupportedError(
    'Interpreter.getInputTensor is not supported on this platform',
  );

  /// Gets the output Tensor for the provided output index.
  Tensor getOutputTensor(int index) => throw UnsupportedError(
    'Interpreter.getOutputTensor is not supported on this platform',
  );

  /// Gets index of an input given the op name of the input.
  int getInputIndex(String opName) => throw UnsupportedError(
    'Interpreter.getInputIndex is not supported on this platform',
  );

  /// Gets index of an output given the op name of the output.
  int getOutputIndex(String opName) => throw UnsupportedError(
    'Interpreter.getOutputIndex is not supported on this platform',
  );

  /// Sets every variable tensor to zero on native platforms; it does not
  /// restore values from the `.tflite` file. Throws [UnsupportedError] here.
  void resetVariableTensors() => throw UnsupportedError(
    'Interpreter.resetVariableTensors is not supported on this platform',
  );

  int getVariableTensorCount() => throw UnsupportedError(
    'Interpreter.getVariableTensorCount is not supported on this platform',
  );

  Tensor getVariableTensor(int index) => throw UnsupportedError(
    'Interpreter.getVariableTensor is not supported on this platform',
  );

  /// Returns the address to the interpreter
  int get address => throw UnsupportedError(
    'Interpreter.address is not supported on this platform',
  );

  bool get isAllocated => throw UnsupportedError(
    'Interpreter.isAllocated is not supported on this platform',
  );

  bool get isDeleted => throw UnsupportedError(
    'Interpreter.isDeleted is not supported on this platform',
  );

  /// Returns the number of signatures defined in the model.
  int get signatureCount => throw UnsupportedError(
    'Interpreter.signatureCount is not supported on this platform',
  );

  /// Returns the key of the signature at [index].
  String getSignatureKey(int index) => throw UnsupportedError(
    'Interpreter.getSignatureKey is not supported on this platform',
  );

  /// Returns the keys of all signatures defined in the model.
  List<String> get signatureKeys => throw UnsupportedError(
    'Interpreter.signatureKeys is not supported on this platform',
  );

  /// Returns a [SignatureRunner] for the given [signatureKey].
  SignatureRunner getSignatureRunner(String signatureKey) =>
      throw UnsupportedError(
        'Interpreter.getSignatureRunner is not supported on this platform',
      );
}
