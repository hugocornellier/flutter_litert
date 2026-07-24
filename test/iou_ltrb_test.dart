import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// [iouLTRB] is the exact intersection-over-union used for frame-to-frame track
/// matching. It must stay distinct from the NMS ratio in `nms_utils.dart`,
/// which adds `1e-7` to the denominator.
void main() {
  group('iouLTRB', () {
    test('identical boxes give 1.0 exactly', () {
      expect(iouLTRB(0, 0, 10, 10, 0, 0, 10, 10), 1.0);
    });

    test('disjoint boxes give 0.0', () {
      expect(iouLTRB(0, 0, 10, 10, 20, 20, 30, 30), 0.0);
    });

    test('edge-touching boxes give 0.0', () {
      expect(iouLTRB(0, 0, 10, 10, 10, 0, 20, 10), 0.0);
    });

    test('half overlap', () {
      // a = 10x10 at origin, b shifted right by 5: intersection 5x10 = 50,
      // union = 100 + 100 - 50 = 150.
      expect(iouLTRB(0, 0, 10, 10, 5, 0, 15, 10), closeTo(50 / 150, 1e-12));
    });

    test('contained box', () {
      // inner 5x5 = 25 inside outer 10x10 = 100; union = 100.
      expect(iouLTRB(0, 0, 10, 10, 0, 0, 5, 5), closeTo(25 / 100, 1e-12));
    });

    test('degenerate (zero-area) boxes give 0.0, not NaN', () {
      expect(iouLTRB(0, 0, 0, 0, 0, 0, 0, 0), 0.0);
      expect(iouLTRB(5, 5, 5, 5, 0, 0, 10, 10), 0.0);
    });

    test('inverted boxes are treated as empty rather than negative', () {
      expect(iouLTRB(10, 10, 0, 0, 0, 0, 10, 10), 0.0);
    });

    test('is exact, with no epsilon in the denominator', () {
      // Union is exactly 150 here. The NMS helpers would divide by 150 + 1e-7
      // and land just below this value; the matcher must not.
      final double exact = 50.0 / 150.0;
      expect(iouLTRB(0, 0, 10, 10, 5, 0, 15, 10), exact);
      expect(iouLTRB(0, 0, 10, 10, 5, 0, 15, 10), isNot(50.0 / (150.0 + 1e-7)));
    });

    test('is symmetric', () {
      final ab = iouLTRB(1, 2, 9, 8, 3, 4, 12, 10);
      final ba = iouLTRB(3, 4, 12, 10, 1, 2, 9, 8);
      expect(ab, ba);
    });
  });
}
