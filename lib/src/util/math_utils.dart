import 'dart:math' as math;

/// Sigmoid activation function.
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

/// Sigmoid with input clipping to prevent overflow.
double sigmoidClipped(double x, {double limit = 80.0}) =>
    sigmoid(clip(x, -limit, limit));

/// Clamps [v] to the range `0.0..1.0`. Returns 0.0 for NaN inputs.
double clamp01(double v) =>
    v.isNaN ? 0.0 : (v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));

/// Clamps [v] to the range `lo..hi`.
double clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

/// Returns indices that sort [a] in descending order.
List<int> argSortDesc(List<double> a) {
  final List<int> idx = List<int>.generate(a.length, (i) => i);
  idx.sort((i, j) => a[j].compareTo(a[i]));
  return idx;
}

/// Returns the median of a non-empty list.
double median(List<double> a) {
  if (a.isEmpty) return double.nan;

  final List<double> b = List<double>.from(a)..sort();
  final int n = b.length;
  if (n.isOdd) return b[n ~/ 2];

  return 0.5 * (b[n ~/ 2 - 1] + b[n ~/ 2]);
}

/// Normalizes an angle in radians to the range `-pi..pi`.
double normalizeRadians(double angle) {
  return angle - 2 * math.pi * ((angle + math.pi) / (2 * math.pi)).floor();
}

/// Exact intersection-over-union of two axis-aligned boxes in left/top/right/
/// bottom coordinates. Returns 0.0 when the boxes are degenerate or disjoint.
///
/// This is the plain `intersection / union` ratio with no epsilon. It is
/// deliberately NOT the ratio used by the NMS helpers in `nms_utils.dart`,
/// which add `1e-7` to the denominator for numerical safety while suppressing
/// candidates. Frame-to-frame track matching compares IoU against a threshold
/// directly, so an epsilon there would shift matches at threshold boundaries.
/// Keep the two separate.
double iouLTRB(
  double aLeft,
  double aTop,
  double aRight,
  double aBottom,
  double bLeft,
  double bTop,
  double bRight,
  double bBottom,
) {
  final double l = math.max(aLeft, bLeft);
  final double t = math.max(aTop, bTop);
  final double r = math.min(aRight, bRight);
  final double b = math.min(aBottom, bBottom);
  final double iw = math.max(0.0, r - l);
  final double ih = math.max(0.0, b - t);
  final double inter = iw * ih;
  final double areaA =
      math.max(0.0, aRight - aLeft) * math.max(0.0, aBottom - aTop);
  final double areaB =
      math.max(0.0, bRight - bLeft) * math.max(0.0, bBottom - bTop);
  final double union = areaA + areaB - inter;
  if (union <= 0) return 0.0;
  return inter / union;
}
