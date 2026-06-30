// Unsupported platform stub for Tensor.
//
// This file is used when compiling for platforms where neither dart:io
// nor dart:js_interop is available.

import 'dart:typed_data';

import '../quantization_params.dart';
import '../tensor_type.dart';

export '../tensor_type.dart';

/// LiteRT tensor.
class Tensor {
  Tensor(dynamic tensor) {
    throw UnsupportedError('Tensor is not supported on this platform');
  }

  /// Name of the tensor element.
  String get name =>
      throw UnsupportedError('Tensor.name is not supported on this platform');

  /// Data type of the tensor element.
  TensorType get type =>
      throw UnsupportedError('Tensor.type is not supported on this platform');

  /// Dimensions of the tensor.
  List<int> get shape =>
      throw UnsupportedError('Tensor.shape is not supported on this platform');

  /// Underlying data buffer as bytes.
  Uint8List get data =>
      throw UnsupportedError('Tensor.data is not supported on this platform');

  /// Mutable float32 view of the tensor buffer.
  Float32List asFloat32View() => throw UnsupportedError(
    'Tensor.asFloat32View is not supported on this platform',
  );

  /// Quantization params associated with the tensor.
  QuantizationParams get params =>
      throw UnsupportedError('Tensor.params is not supported on this platform');

  /// Updates the underlying data buffer with new bytes.
  set data(Uint8List bytes) =>
      throw UnsupportedError('Tensor.data= is not supported on this platform');

  /// Returns number of dimensions
  int numDimensions() => throw UnsupportedError(
    'Tensor.numDimensions is not supported on this platform',
  );

  /// Returns the size, in bytes, of the tensor data.
  int numBytes() => throw UnsupportedError(
    'Tensor.numBytes is not supported on this platform',
  );

  /// Returns the number of elements in a flattened (1-D) view of the tensor.
  int numElements() => throw UnsupportedError(
    'Tensor.numElements is not supported on this platform',
  );

  void setTo(Object src) =>
      throw UnsupportedError('Tensor.setTo is not supported on this platform');

  Object copyTo(Object dst) =>
      throw UnsupportedError('Tensor.copyTo is not supported on this platform');

  List<int>? getInputShapeIfDifferent(Object? input) => throw UnsupportedError(
    'Tensor.getInputShapeIfDifferent is not supported on this platform',
  );

  @override
  String toString() => 'Tensor(unsupported platform stub)';
}
