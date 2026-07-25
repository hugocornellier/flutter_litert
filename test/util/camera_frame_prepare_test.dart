import 'dart:typed_data';

import 'package:flutter/foundation.dart'
    show debugDefaultTargetPlatformOverride, TargetPlatform;
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Coverage for the camera-frame entry points that detector packages actually
/// call: [prepareCameraFrame], [prepareCameraFrameFromImage] and
/// [rotationForFrame].
///
/// The layers underneath these (`packYuv420`, `PackedImageLayout`,
/// `coverFitScaleOffset`) already have their own tests; this file covers the
/// dispatch layer on top, where plane shape is inspected to pick a layout and
/// where camera geometry is turned into a rotation.
void main() {
  /// A plane with `pixelStride` bytes per pixel and `rowStride` bytes per row.
  CameraPlane plane({
    required int rowStride,
    required int pixelStride,
    required int length,
    int fill = 0,
  }) => (
    bytes: Uint8List(length)..fillRange(0, length, fill),
    rowStride: rowStride,
    pixelStride: pixelStride,
  );

  group('prepareCameraFrame - packed BGRA/RGBA (single plane)', () {
    test('maps a tightly packed BGRA plane to bgra2bgr', () {
      final p = plane(rowStride: 4 * 4, pixelStride: 4, length: 4 * 4 * 2);
      final frame = prepareCameraFrame(width: 4, height: 2, planes: [p]);

      expect(frame, isNotNull);
      expect(frame!.conversion, CameraFrameConversion.bgra2bgr);
      expect(frame.width, 4);
      expect(frame.height, 2);
      expect(frame.strideCols, 4, reason: 'rowStride ~/ 4 with no padding');
      expect(frame.rotation, isNull);
    });

    test('isBgra: false selects rgba2bgr instead', () {
      final frame = prepareCameraFrame(
        width: 4,
        height: 2,
        planes: [plane(rowStride: 16, pixelStride: 4, length: 32)],
        isBgra: false,
      );

      expect(frame!.conversion, CameraFrameConversion.rgba2bgr);
    });

    test('references the source buffer without copying', () {
      // The doc contract is "no copy" for this path: detector packages rely on
      // it to avoid a full-frame allocation per camera frame.
      final p = plane(rowStride: 16, pixelStride: 4, length: 32);
      final frame = prepareCameraFrame(width: 4, height: 2, planes: [p]);

      expect(identical(frame!.bytes, p.bytes), isTrue);
    });

    test('keeps row-stride padding in strideCols, not width', () {
      // A 5-pixel-wide frame padded to 8 columns: the consumer needs both the
      // padded column count (to reconstruct the Mat) and the visible width
      // (to crop). Collapsing them would shear the image.
      final frame = prepareCameraFrame(
        width: 5,
        height: 3,
        planes: [plane(rowStride: 8 * 4, pixelStride: 4, length: 8 * 4 * 3)],
      );

      expect(frame!.width, 5);
      expect(frame.strideCols, 8);
      expect(frame.decodePlan().hasStridePadding, isTrue);
    });

    test('passes rotation through unchanged', () {
      for (final r in CameraFrameRotation.values) {
        final frame = prepareCameraFrame(
          width: 4,
          height: 2,
          planes: [plane(rowStride: 16, pixelStride: 4, length: 32)],
          rotation: r,
        );
        expect(frame!.rotation, r);
      }
    });
  });

  group('prepareCameraFrame - YUV420', () {
    // 4x2 frame: 8 luma bytes, 2x1 chroma.
    CameraPlane yPlane() =>
        plane(rowStride: 4, pixelStride: 1, length: 8, fill: 16);

    test('two planes are treated as NV12', () {
      final frame = prepareCameraFrame(
        width: 4,
        height: 2,
        planes: [
          yPlane(),
          plane(rowStride: 4, pixelStride: 2, length: 4, fill: 128),
        ],
      );

      expect(frame, isNotNull);
      expect(frame!.conversion, CameraFrameConversion.yuv2bgrNv12);
      expect(
        frame.strideCols,
        frame.width,
        reason: 'packYuv420 output is tightly packed, so no padding',
      );
    });

    test('three planes with chroma pixelStride 2 are NV21', () {
      final frame = prepareCameraFrame(
        width: 4,
        height: 2,
        planes: [
          yPlane(),
          plane(rowStride: 4, pixelStride: 2, length: 4, fill: 100),
          plane(rowStride: 4, pixelStride: 2, length: 4, fill: 200),
        ],
      );

      expect(frame!.conversion, CameraFrameConversion.yuv2bgrNv21);
    });

    test('three planes with chroma pixelStride 1 are I420', () {
      final frame = prepareCameraFrame(
        width: 4,
        height: 2,
        planes: [
          yPlane(),
          plane(rowStride: 2, pixelStride: 1, length: 2, fill: 100),
          plane(rowStride: 2, pixelStride: 1, length: 2, fill: 200),
        ],
      );

      expect(frame!.conversion, CameraFrameConversion.yuv2bgrI420);
    });

    test('odd dimensions are rejected', () {
      // YUV420 subsamples chroma by 2, so odd dimensions have no valid packing.
      final frame = prepareCameraFrame(
        width: 5,
        height: 3,
        planes: [
          plane(rowStride: 5, pixelStride: 1, length: 15),
          plane(rowStride: 5, pixelStride: 2, length: 8),
        ],
      );

      expect(frame, isNull);
    });
  });

  group('prepareCameraFrame - unsupported shapes return null', () {
    test('empty plane list', () {
      expect(prepareCameraFrame(width: 4, height: 2, planes: []), isNull);
    });

    test('single plane that is not 4-channel', () {
      // One plane with pixelStride 1 is neither packed RGBA nor a complete
      // YUV frame (no chroma), so there is nothing to decode.
      expect(
        prepareCameraFrame(
          width: 4,
          height: 2,
          planes: [plane(rowStride: 4, pixelStride: 1, length: 8)],
        ),
        isNull,
      );
    });
  });

  group('prepareCameraFrameFromImage', () {
    test('accepts a duck-typed CameraImage and matches prepareCameraFrame', () {
      final image = _FakeCameraImage(
        width: 4,
        height: 2,
        planes: [_FakePlane(Uint8List(32), bytesPerRow: 16, bytesPerPixel: 4)],
      );

      final viaImage = prepareCameraFrameFromImage(image, isBgra: true);
      final direct = prepareCameraFrame(
        width: 4,
        height: 2,
        planes: [(bytes: image.planes[0].bytes, rowStride: 16, pixelStride: 4)],
      );

      expect(viaImage, isNotNull);
      expect(viaImage!.conversion, direct!.conversion);
      expect(viaImage.width, direct.width);
      expect(viaImage.height, direct.height);
      expect(viaImage.strideCols, direct.strideCols);
    });

    test('a null bytesPerPixel defaults to a pixel stride of 1', () {
      // package:camera reports a null bytesPerPixel for YUV planes on Android.
      final image = _FakeCameraImage(
        width: 4,
        height: 2,
        planes: [
          _FakePlane(Uint8List(8), bytesPerRow: 4, bytesPerPixel: null),
          _FakePlane(Uint8List(4), bytesPerRow: 4, bytesPerPixel: null),
        ],
      );

      // Stride 1 on both planes means the two-plane path, i.e. NV12, rather
      // than being misread as a packed 4-channel buffer.
      final frame = prepareCameraFrameFromImage(image);
      expect(frame, isNotNull);
      expect(frame!.conversion, CameraFrameConversion.yuv2bgrNv12);
    });

    test('defaults isBgra from the host platform', () {
      final image = _FakeCameraImage(
        width: 4,
        height: 2,
        planes: [_FakePlane(Uint8List(32), bytesPerRow: 16, bytesPerPixel: 4)],
      );

      debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
      expect(
        prepareCameraFrameFromImage(image)!.conversion,
        CameraFrameConversion.bgra2bgr,
        reason: 'macOS camera_desktop delivers BGRA',
      );

      debugDefaultTargetPlatformOverride = TargetPlatform.linux;
      expect(
        prepareCameraFrameFromImage(image)!.conversion,
        CameraFrameConversion.rgba2bgr,
        reason: 'Linux camera_desktop delivers RGBA',
      );

      debugDefaultTargetPlatformOverride = null;
    });

    test('an explicit isBgra overrides the platform default', () {
      final image = _FakeCameraImage(
        width: 4,
        height: 2,
        planes: [_FakePlane(Uint8List(32), bytesPerRow: 16, bytesPerPixel: 4)],
      );

      debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
      expect(
        prepareCameraFrameFromImage(image, isBgra: false)!.conversion,
        CameraFrameConversion.rgba2bgr,
      );
      debugDefaultTargetPlatformOverride = null;
    });

    test('throws on an object without the expected shape', () {
      // Documented tradeoff: duck typing fails at runtime rather than compile
      // time. Pinned so the failure stays a throw and never a silent null.
      expect(
        () => prepareCameraFrameFromImage(Object()),
        throwsA(isA<NoSuchMethodError>()),
      );
    });
  });

  group('rotationForFrame - iOS', () {
    setUp(() => debugDefaultTargetPlatformOverride = TargetPlatform.iOS);
    tearDown(() => debugDefaultTargetPlatformOverride = null);

    CameraFrameRotation? rotate({
      int width = 1920,
      int height = 1080,
      int sensorOrientation = 90,
      bool isFrontCamera = false,
      DeviceOrientation deviceOrientation = DeviceOrientation.portraitUp,
    }) => rotationForFrame(
      width: width,
      height: height,
      sensorOrientation: sensorOrientation,
      isFrontCamera: isFrontCamera,
      deviceOrientation: deviceOrientation,
    );

    test('portrait with a landscape buffer rotates by sensor orientation', () {
      expect(rotate(sensorOrientation: 90), CameraFrameRotation.cw90);
      expect(rotate(sensorOrientation: 270), CameraFrameRotation.cw270);
      expect(rotate(sensorOrientation: 0), isNull);
      expect(rotate(sensorOrientation: 180), isNull);
    });

    test('portraitDown is still portrait', () {
      expect(
        rotate(deviceOrientation: DeviceOrientation.portraitDown),
        CameraFrameRotation.cw90,
      );
    });

    test('landscape device orientations need no rotation', () {
      expect(
        rotate(deviceOrientation: DeviceOrientation.landscapeLeft),
        isNull,
      );
      expect(
        rotate(deviceOrientation: DeviceOrientation.landscapeRight),
        isNull,
      );
    });

    test('an already-portrait buffer is not rotated again', () {
      expect(rotate(width: 1080, height: 1920), isNull);
      expect(
        rotate(width: 1080, height: 1080),
        isNull,
        reason: 'square counts as height >= width',
      );
    });
  });

  group('rotationForFrame - Android', () {
    setUp(() => debugDefaultTargetPlatformOverride = TargetPlatform.android);
    tearDown(() => debugDefaultTargetPlatformOverride = null);

    CameraFrameRotation? rotate({
      required int sensorOrientation,
      required bool isFrontCamera,
      required DeviceOrientation deviceOrientation,
    }) => rotationForFrame(
      width: 1920,
      height: 1080,
      sensorOrientation: sensorOrientation,
      isFrontCamera: isFrontCamera,
      deviceOrientation: deviceOrientation,
    );

    test('back camera subtracts device rotation from sensor orientation', () {
      // sensor 90, portraitUp (0) -> 90
      expect(
        rotate(
          sensorOrientation: 90,
          isFrontCamera: false,
          deviceOrientation: DeviceOrientation.portraitUp,
        ),
        CameraFrameRotation.cw90,
      );
      // sensor 90, landscapeLeft (90) -> 0, i.e. no rotation
      expect(
        rotate(
          sensorOrientation: 90,
          isFrontCamera: false,
          deviceOrientation: DeviceOrientation.landscapeLeft,
        ),
        isNull,
      );
      // sensor 90, portraitDown (180) -> 270 (wraps via +360)
      expect(
        rotate(
          sensorOrientation: 90,
          isFrontCamera: false,
          deviceOrientation: DeviceOrientation.portraitDown,
        ),
        CameraFrameRotation.cw270,
      );
      // sensor 90, landscapeRight (270) -> 180
      expect(
        rotate(
          sensorOrientation: 90,
          isFrontCamera: false,
          deviceOrientation: DeviceOrientation.landscapeRight,
        ),
        CameraFrameRotation.cw180,
      );
    });

    test('front camera adds device rotation instead of subtracting', () {
      // The mirrored sensor is why the sign flips; getting this backwards
      // shows up as upside-down selfie preview.
      expect(
        rotate(
          sensorOrientation: 270,
          isFrontCamera: true,
          deviceOrientation: DeviceOrientation.portraitUp,
        ),
        CameraFrameRotation.cw270,
      );
      expect(
        rotate(
          sensorOrientation: 270,
          isFrontCamera: true,
          deviceOrientation: DeviceOrientation.landscapeLeft,
        ),
        isNull,
        reason: '(270 + 90) % 360 == 0',
      );
      expect(
        rotate(
          sensorOrientation: 270,
          isFrontCamera: true,
          deviceOrientation: DeviceOrientation.portraitDown,
        ),
        CameraFrameRotation.cw90,
      );
    });

    test('front and back disagree except where the arithmetic coincides', () {
      final back = rotate(
        sensorOrientation: 90,
        isFrontCamera: false,
        deviceOrientation: DeviceOrientation.landscapeLeft,
      );
      final front = rotate(
        sensorOrientation: 90,
        isFrontCamera: true,
        deviceOrientation: DeviceOrientation.landscapeLeft,
      );

      expect(back, isNull, reason: '(90 - 90) == 0');
      expect(front, CameraFrameRotation.cw180, reason: '(90 + 90) == 180');
    });
  });

  group('rotationForFrame - other platforms', () {
    tearDown(() => debugDefaultTargetPlatformOverride = null);

    test('desktop platforms never rotate', () {
      for (final p in [
        TargetPlatform.macOS,
        TargetPlatform.linux,
        TargetPlatform.windows,
      ]) {
        debugDefaultTargetPlatformOverride = p;
        expect(
          rotationForFrame(
            width: 1920,
            height: 1080,
            sensorOrientation: 90,
            isFrontCamera: false,
            deviceOrientation: DeviceOrientation.portraitUp,
          ),
          isNull,
          reason: '$p delivers upright frames',
        );
      }
    });
  });
}

class _FakePlane {
  _FakePlane(
    this.bytes, {
    required this.bytesPerRow,
    required this.bytesPerPixel,
  });

  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;
}

class _FakeCameraImage {
  _FakeCameraImage({
    required this.width,
    required this.height,
    required this.planes,
  });

  final int width;
  final int height;
  final List<_FakePlane> planes;
}
