import 'dart:math' as math;

import '../compiled_model/compiled_model.dart';

/// Helpers for deriving tensor geometry from a [CompiledModel].
///
/// The LiteRT Next [CompiledModel] engine exposes tensor sizes only in bytes
/// (there is no `Interpreter` to query shapes from), so every detector that
/// uses the compiled engine re-derives float counts and the square input side
/// from `inputByteSizes` / `outputByteSizes` with the same `byteSize ~/ 4`
/// arithmetic. These helpers centralize that math so the `~/ 4` conversion and
/// the perfect-square check live in one tested place.
///
/// Output *role* resolution (which output is landmarks vs score vs boxes) stays
/// in each model because it is genuinely model-specific; the float-count and
/// square-side primitives below are shared. They throw [UnsupportedError] on a
/// mismatch (the same error type the detectors already throw, so the
/// compiled-engine→Interpreter-engine fallback is preserved).

/// Number of float32 values in a tensor of [byteSize] bytes.
///
/// Throws [UnsupportedError] if [byteSize] is not a positive multiple of 4.
int compiledFloatCount(int byteSize, {String label = 'tensor'}) {
  if (byteSize <= 0 || byteSize % 4 != 0) {
    throw UnsupportedError(
      'Compiled $label byte size $byteSize is not a positive multiple of 4 '
      '(expected float32 data).',
    );
  }
  return byteSize ~/ 4;
}

/// The side length `S` of a square `[1, S, S, channels]` tensor that holds
/// [floats] float32 values.
///
/// Throws [UnsupportedError] if [floats] is not `S*S*channels` for a positive
/// integer `S`.
int squareSideFromFloats(
  int floats, {
  int channels = 3,
  String label = 'input',
}) {
  if (channels <= 0 || floats <= 0 || floats % channels != 0) {
    throw UnsupportedError(
      'Compiled $label has $floats floats, not a positive multiple of '
      '$channels channels.',
    );
  }
  final int area = floats ~/ channels;
  final int side = math.sqrt(area).round();
  if (side * side != area) {
    throw UnsupportedError(
      'Compiled $label area $area is not a perfect square '
      '(expected [1, S, S, $channels]).',
    );
  }
  return side;
}

/// The square input side (e.g. 192 or 224) of a single-input [model], derived
/// from its input byte size.
///
/// Throws [UnsupportedError] if the model does not have exactly one input or
/// the input is not a square `[1, S, S, channels]` float32 tensor.
int compiledSquareInputSide(
  CompiledModel model, {
  int channels = 3,
  String label = 'model',
}) {
  if (model.inputCount != 1) {
    throw UnsupportedError(
      'Compiled $label expects one input tensor; got ${model.inputCount}.',
    );
  }
  final int floats = compiledFloatCount(
    model.inputByteSizes.single,
    label: '$label input',
  );
  return squareSideFromFloats(
    floats,
    channels: channels,
    label: '$label input',
  );
}

/// Float counts for every output tensor of [model], in order.
List<int> compiledOutputFloatCounts(
  CompiledModel model, {
  String label = 'model',
}) {
  final List<int> sizes = model.outputByteSizes;
  return <int>[
    for (int i = 0; i < sizes.length; i++)
      compiledFloatCount(sizes[i], label: '$label output $i'),
  ];
}

/// Index of the first entry in [floatCounts] for which [test] returns true, or
/// `-1` if none match.
int indexWhereFloatCount(
  List<int> floatCounts,
  bool Function(int floats) test,
) {
  for (int i = 0; i < floatCounts.length; i++) {
    if (test(floatCounts[i])) return i;
  }
  return -1;
}
