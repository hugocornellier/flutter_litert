import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('CoverFitTransform', () {
    test('wide source into square viewport covers height and centers x', () {
      // source 200x100 (aspect 2.0) into viewport 100x100 (aspect 1.0):
      // sourceAspect > viewAspect -> scale by height, center on x.
      final t = CoverFitTransform.cover(
        sourceWidth: 200,
        sourceHeight: 100,
        viewWidth: 100,
        viewHeight: 100,
      );
      expect(t.scale, closeTo(1.0, 1e-9)); // 100 / 100
      expect(t.offsetX, closeTo(-50.0, 1e-9)); // (100 - 200) / 2
      expect(t.offsetY, closeTo(0.0, 1e-9));
    });

    test('map applies scale then offset', () {
      final t = CoverFitTransform.cover(
        sourceWidth: 200,
        sourceHeight: 100,
        viewWidth: 100,
        viewHeight: 100,
      );
      final p = t.map(100, 50); // (100 * 1 - 50, 50 * 1 + 0)
      expect(p.dx, closeTo(50, 1e-9));
      expect(p.dy, closeTo(50, 1e-9));
    });

    test('mirror reflects x about sourceWidth before scaling', () {
      final t = CoverFitTransform.cover(
        sourceWidth: 200,
        sourceHeight: 100,
        viewWidth: 100,
        viewHeight: 100,
        mirror: true,
      );
      // x=0 mirrors to 200 -> 200 * 1 - 50 = 150
      expect(t.map(0, 0).dx, closeTo(150, 1e-9));
      // x=200 mirrors to 0 -> 0 * 1 - 50 = -50
      expect(t.map(200, 0).dx, closeTo(-50, 1e-9));
    });

    test('scaleLength multiplies a length by the uniform scale', () {
      final t = CoverFitTransform.cover(
        sourceWidth: 100,
        sourceHeight: 200,
        viewWidth: 100,
        viewHeight: 100,
      );
      expect(t.scale, closeTo(1.0, 1e-9));
      expect(t.scaleLength(4), closeTo(4.0, 1e-9));
    });

    test('matches coverFitScaleOffset for the non-mirrored mapping', () {
      final fit = coverFitScaleOffset(640, 480, 300.0, 300.0);
      final t = CoverFitTransform.cover(
        sourceWidth: 640,
        sourceHeight: 480,
        viewWidth: 300,
        viewHeight: 300,
      );
      expect(t.scale, closeTo(fit.scale, 1e-9));
      final p = t.map(320, 240);
      expect(p.dx, closeTo(320 * fit.scale + fit.offsetX, 1e-9));
      expect(p.dy, closeTo(240 * fit.scale + fit.offsetY, 1e-9));
    });
  });
}
