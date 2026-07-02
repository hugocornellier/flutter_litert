/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import '../bindings/bindings.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';
import '../util/byte_conversion_utils_native.dart';

import '../ffi/helper.dart';
import '../quantization_params.dart';
import '../tensor_type.dart';
import '../util/list_utils.dart' as list_utils;
import '../util/tensor_shape_utils.dart' as shape_utils;

export '../bindings/tensorflow_lite_bindings_generated.dart' show TfLiteType;
export '../tensor_type.dart';

/// LiteRT tensor.
class Tensor {
  final Pointer<TfLiteTensor> _tensor;

  /// Creates a tensor wrapper around a native tensor pointer.
  Tensor(this._tensor) {
    ArgumentError.checkNotNull(_tensor);
  }

  /// Name of the tensor element.
  String get name =>
      tfliteBinding.TfLiteTensorName(_tensor).cast<Utf8>().toDartString();

  /// Data type of the tensor element.
  TensorType get type =>
      TensorType.fromValue(tfliteBinding.TfLiteTensorType(_tensor));

  /// Dimensions of the tensor.
  List<int> get shape => List.generate(
    tfliteBinding.TfLiteTensorNumDims(_tensor),
    (i) => tfliteBinding.TfLiteTensorDim(_tensor, i),
  );

  /// Underlying data buffer as bytes.
  Uint8List get data {
    final data = cast<Uint8>(tfliteBinding.TfLiteTensorData(_tensor));
    return data
        .asTypedList(tfliteBinding.TfLiteTensorByteSize(_tensor))
        .asUnmodifiableView();
  }

  /// Mutable [Float32List] view aliasing this tensor's native buffer.
  ///
  /// Writes go directly into tensor memory with no copy; both indexed
  /// stores and bulk [Float32List.setAll]/[Float32List.setRange] are
  /// supported. The view is valid until the next
  /// `resizeInputTensor`/`allocateTensors` call; recapture after either.
  ///
  /// The bytes are reinterpreted as float32 regardless of the tensor's
  /// element type; for quantized tensors use [data] instead.
  Float32List asFloat32View() {
    final byteSize = tfliteBinding.TfLiteTensorByteSize(_tensor);
    if (byteSize % 4 != 0) {
      throw ArgumentError('Tensor byte size $byteSize is not float32-aligned.');
    }
    final data = cast<Float>(tfliteBinding.TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    return data.asTypedList(byteSize ~/ 4);
  }

  /// Quantization params associated with the tensor.
  QuantizationParams get params {
    final ref = tfliteBinding.TfLiteTensorQuantizationParams(_tensor);
    return QuantizationParams(ref.scale, ref.zero_point);
  }

  /// Updates the underlying data buffer with new bytes.
  ///
  /// The size must match the size of the tensor.
  set data(Uint8List bytes) {
    final tensorByteSize = tfliteBinding.TfLiteTensorByteSize(_tensor);
    checkArgument(tensorByteSize == bytes.length);
    final data = cast<Uint8>(tfliteBinding.TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    final externalTypedData = data.asTypedList(tensorByteSize);
    externalTypedData.setRange(0, tensorByteSize, bytes);
  }

  /// Returns number of dimensions
  int numDimensions() {
    return tfliteBinding.TfLiteTensorNumDims(_tensor);
  }

  /// Returns the size, in bytes, of the tensor data.
  int numBytes() {
    return tfliteBinding.TfLiteTensorByteSize(_tensor);
  }

  /// Returns the number of elements in a flattened (1-D) view of the tensor.
  int numElements() {
    return shape_utils.computeNumElements(shape);
  }

  /// Copies the given [src] data into this tensor.
  void setTo(Object src) {
    Uint8List bytes = _convertObjectToBytes(src);
    int size = bytes.length;

    // String tensors require buffer reallocation because the pre-allocated
    // buffer size does not match the encoded string data size. The native
    // copy path is kept here because the data pointer moves on realloc.
    if (type == TensorType.string) {
      final reallocStatus = tfliteBinding.TfLiteTensorRealloc(size, _tensor);
      checkState(
        reallocStatus == TfLiteStatus.kTfLiteOk,
        message:
            'TfLiteTensorRealloc failed for string tensor '
            '(requested $size bytes, status=$reallocStatus).',
      );
      final ptr = calloc<Uint8>(size);
      checkState(isNotNull(ptr), message: 'unallocated');
      ptr.asTypedList(size).setRange(0, size, bytes);
      try {
        checkState(
          tfliteBinding.TfLiteTensorCopyFromBuffer(_tensor, ptr.cast(), size) ==
              TfLiteStatus.kTfLiteOk,
          message:
              'TfLiteTensorCopyFromBuffer failed '
              '(buffer=$size bytes, tensor=${numBytes()} bytes).',
        );
      } finally {
        calloc.free(ptr);
      }
      return;
    }

    // Single memcpy straight into the tensor's buffer; no native scratch copy.
    // Plain if+throw instead of checkState: quiver builds the interpolated
    // message eagerly, on every successful call in the hot path.
    final tensorByteSize = tfliteBinding.TfLiteTensorByteSize(_tensor);
    if (tensorByteSize != size) {
      throw StateError(
        'TfLiteTensorCopyFromBuffer failed '
        '(buffer=$size bytes, tensor=$tensorByteSize bytes).',
      );
    }
    final data = cast<Uint8>(tfliteBinding.TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    data.asTypedList(tensorByteSize).setRange(0, tensorByteSize, bytes);
  }

  /// Copies this tensor's data into [dst].
  ///
  /// When [dst] is typed data matching the tensor's element type
  /// (e.g. [Float32List] for a float32 tensor), the bytes are bulk-copied
  /// directly and [dst] itself is returned; no per-element conversion or
  /// nested-list allocation.
  Object copyTo(Object dst) {
    int size = tfliteBinding.TfLiteTensorByteSize(_tensor);
    final data = cast<Uint8>(tfliteBinding.TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    // View over native memory: valid only until the next invoke/allocate.
    // Every branch below copies out of it before returning.
    final src = data.asTypedList(size);

    final tensorType = type;
    if (dst is Uint8List) {
      if (dst.length != size) {
        throw ArgumentError(
          'Output object shape mismatch: tensor has $size bytes but the '
          'provided Uint8List has ${dst.length}.',
        );
      }
      dst.setAll(0, src);
      return dst;
    }
    if (dst is ByteBuffer) {
      final view = dst.asUint8List();
      view.setRange(0, view.length, src);
      return dst;
    }
    final typedDst = _copyToTypedData(dst, src, tensorType, size);
    if (typedDst != null) {
      return typedDst;
    }

    final obj = _convertBytesToObject(src);
    if (obj is List && dst is List) {
      list_utils.duplicateList(obj, dst);
    }
    return obj;
  }

  /// Bulk-copies tensor bytes into [dst] when its runtime type matches
  /// [tensorType]. Returns null when no fast path applies.
  Object? _copyToTypedData(
    Object dst,
    Uint8List src,
    TensorType tensorType,
    int size,
  ) {
    final TypedData? view;
    if (dst is Float32List && tensorType == TensorType.float32) {
      view = src.buffer.asFloat32List(src.offsetInBytes, size ~/ 4);
    } else if (dst is Int32List && tensorType == TensorType.int32) {
      view = src.buffer.asInt32List(src.offsetInBytes, size ~/ 4);
    } else if (dst is Int64List && tensorType == TensorType.int64) {
      view = src.buffer.asInt64List(src.offsetInBytes, size ~/ 8);
    } else if (dst is Int16List && tensorType == TensorType.int16) {
      view = src.buffer.asInt16List(src.offsetInBytes, size ~/ 2);
    } else if (dst is Int8List && tensorType == TensorType.int8) {
      view = src.buffer.asInt8List(src.offsetInBytes, size);
    } else {
      return null;
    }
    final typed = view as List;
    final out = dst as List;
    if (out.length != typed.length) {
      throw ArgumentError(
        'Output object shape mismatch: tensor has ${typed.length} '
        '$tensorType elements but the provided ${dst.runtimeType} '
        'has ${out.length}.',
      );
    }
    out.setAll(0, typed);
    return dst;
  }

  Uint8List _convertObjectToBytes(Object o) {
    return ByteConversionUtils.convertObjectToBytes(o, type);
  }

  Object _convertBytesToObject(Uint8List bytes) {
    return ByteConversionUtils.convertBytesToObject(bytes, type, shape);
  }

  List<int>? getInputShapeIfDifferent(Object? input) {
    if (input == null) return null;
    if (input is ByteBuffer || input is Uint8List) return null;
    if (input is TypedData && input is List) {
      // Same decision as list_utils.getInputShapeIfDifferent for flat typed
      // data, but derives the element count from the tensor's byte size
      // instead of materializing the shape list, which costs NumDims plus
      // one Dim FFI call per axis on every run.
      final elementSize = _fixedElementByteSize(type);
      if (elementSize > 0) {
        final length = (input as List).length;
        if (length * elementSize ==
            tfliteBinding.TfLiteTensorByteSize(_tensor)) {
          return null;
        }
        return [length];
      }
    }
    return list_utils.getInputShapeIfDifferent(input, shape);
  }

  @override
  String toString() {
    return 'Tensor{_tensor: $_tensor, name: $name, type: $type, shape: $shape, data: ${data.length}}';
  }
}

/// Bytes per element for fixed-width tensor types, or -1 when the element
/// width is not a per-element constant (strings, packed int4, resources).
int _fixedElementByteSize(TensorType type) {
  switch (type) {
    case TensorType.float32:
    case TensorType.int32:
    case TensorType.uint32:
      return 4;
    case TensorType.float64:
    case TensorType.int64:
    case TensorType.uint64:
    case TensorType.complex64:
      return 8;
    case TensorType.int16:
    case TensorType.uint16:
    case TensorType.float16:
      return 2;
    case TensorType.int8:
    case TensorType.uint8:
    case TensorType.boolean:
      return 1;
    case TensorType.complex128:
      return 16;
    case TensorType.noType:
    case TensorType.string:
    case TensorType.int4:
    case TensorType.resource:
    case TensorType.variant:
      return -1;
  }
}
