import 'dart:typed_data';

import 'list_shape_extension.dart';
import 'tensor_shape_utils.dart' as shape_utils;

/// TFLite type integer constants (matching TfLiteType / TensorType enum values).
const int _kFloat32 = 1;
const int _kInt32 = 2;
const int _kString = 5;
const int _kBool = 6;

/// Returns the TFLite data type integer for the leaf element of [o].
int dataTypeOf(Object o) {
  while (o is List) {
    o = o.elementAt(0);
  }
  if (o is double) return _kFloat32;
  if (o is int) return _kInt32;
  if (o is String) return _kString;
  if (o is bool) return _kBool;
  throw ArgumentError(
    'DataType error: cannot resolve DataType of ${o.runtimeType}',
  );
}

/// Copies elements from [obj] to [dst], verifying shapes match.
void duplicateList(List obj, List dst) {
  var objShape = obj.shape;
  var dstShape = dst.shape;
  var equal = true;
  if (objShape.length == dst.shape.length) {
    for (var i = 0; i < objShape.length; i++) {
      if (objShape[i] != dstShape[i]) {
        equal = false;
        break;
      }
    }
  } else {
    equal = false;
  }
  if (!equal) {
    throw ArgumentError(
      'Output object shape mismatch, interpreter returned output of shape: ${obj.shape} while shape of output provided as argument in run is: ${dst.shape}',
    );
  }
  for (var i = 0; i < obj.length; i++) {
    dst[i] = obj[i];
  }
}

/// Returns the shape of [input] if it differs from [tensorShape], else null.
List<int>? getInputShapeIfDifferent(Object? input, List<int> tensorShape) {
  if (input == null) return null;
  if (input is ByteBuffer || input is Uint8List) return null;
  if (input is TypedData && input is List) {
    // Flat typed data staged into a multi-dimensional tensor: when the
    // element counts match, treat it as already shaped; its 1-D length is
    // a layout, not a resize request. Resizing the tensor to rank 1 would
    // break models whose ops require the original rank.
    final length = (input as List).length;
    if (length == shape_utils.computeNumElements(tensorShape)) return null;
    return [length];
  }
  final inputShape = shape_utils.computeShapeOf(input);
  if (_intListEquals(inputShape, tensorShape)) return null;
  return inputShape;
}

bool _intListEquals(List<int> a, List<int> b) {
  if (identical(a, b)) return true;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
