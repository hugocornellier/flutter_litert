import '../native/interpreter.dart';

/// Collects the shape of every output tensor of [itp], keyed by output index.
///
/// Model wrappers commonly need output shapes at initialization to work out
/// which output carries which result (by element count, trailing dimension, and
/// so on) and to preallocate result buffers. This walks output indices from 0
/// until `getOutputTensor` throws, so it works on interpreters that do not
/// expose a reliable output count.
///
/// Only shapes are read. Deliberately does not touch `Tensor.data`, so no
/// buffer views are materialized and quantized outputs are safe to enumerate.
/// Use [TensorFloat32Views] when you need the tensor buffers themselves.
///
/// [Interpreter.allocateTensors] should have been called first, otherwise the
/// reported shapes may not reflect the allocated tensors.
Map<int, List<int>> collectOutputShapes(Interpreter itp) {
  final Map<int, List<int>> shapes = <int, List<int>>{};
  for (int i = 0; ; i++) {
    try {
      shapes[i] = itp.getOutputTensor(i).shape;
    } catch (_) {
      break;
    }
  }
  return shapes;
}
