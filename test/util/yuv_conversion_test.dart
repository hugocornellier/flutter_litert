import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

void main() {
  group('packYuv420', () {
    test('NV12 (2 planes, no padding): contiguous Y + UV', () {
      const w = 4, h = 4;
      final yBytes = Uint8List.fromList(List<int>.generate(w * h, (i) => i));
      final uvBytes = Uint8List.fromList(
        List<int>.generate(w * (h ~/ 2), (i) => 100 + i),
      );

      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: yBytes, rowStride: w, pixelStride: 1),
        u: (bytes: uvBytes, rowStride: w, pixelStride: 2),
      );

      expect(packed, isNotNull);
      expect(packed!.layout, YuvLayout.nv12);
      expect(packed.width, w);
      expect(packed.height, h);
      expect(packed.bytes.length, w * h + w * (h ~/ 2));
      expect(packed.bytes.sublist(0, w * h), yBytes);
      expect(packed.bytes.sublist(w * h), uvBytes);
    });

    test('NV21 (3 planes, pixelStride 2) uses V buffer as VU region', () {
      const w = 4, h = 4;
      final yBytes = Uint8List.fromList(List<int>.generate(w * h, (i) => i));
      // V plane byte 0 starts the VU-interleaved region (NV21 layout).
      final vuBuf = Uint8List.fromList(
        List<int>.generate(w * (h ~/ 2), (i) => 200 + i),
      );

      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: yBytes, rowStride: w, pixelStride: 1),
        u: (bytes: vuBuf, rowStride: w, pixelStride: 2),
        v: (bytes: vuBuf, rowStride: w, pixelStride: 2),
      );

      expect(packed, isNotNull);
      expect(packed!.layout, YuvLayout.nv21);
      expect(packed.bytes.sublist(w * h), vuBuf);
    });

    test('I420 (3 planes, pixelStride 1) concatenates Y + U + V', () {
      const w = 4, h = 4;
      final yBytes = Uint8List.fromList(List<int>.generate(w * h, (i) => i));
      final uBytes = Uint8List.fromList(
        List<int>.generate((w ~/ 2) * (h ~/ 2), (i) => 50 + i),
      );
      final vBytes = Uint8List.fromList(
        List<int>.generate((w ~/ 2) * (h ~/ 2), (i) => 150 + i),
      );

      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: yBytes, rowStride: w, pixelStride: 1),
        u: (bytes: uBytes, rowStride: w ~/ 2, pixelStride: 1),
        v: (bytes: vBytes, rowStride: w ~/ 2, pixelStride: 1),
      );

      expect(packed, isNotNull);
      expect(packed!.layout, YuvLayout.i420);
      expect(packed.bytes.length, w * h + (w ~/ 2) * (h ~/ 2) * 2);
      expect(packed.bytes.sublist(0, w * h), yBytes);
      expect(packed.bytes.sublist(w * h, w * h + (w ~/ 2) * (h ~/ 2)), uBytes);
      expect(packed.bytes.sublist(w * h + (w ~/ 2) * (h ~/ 2)), vBytes);
    });

    test('strips row-stride padding from Y plane', () {
      const w = 4, h = 2, stride = 8; // 4 bytes padding per row
      final yBuf = Uint8List(stride * h);
      for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
          yBuf[r * stride + c] = r * 10 + c; // padding bytes stay zero
        }
      }
      final uvBuf = Uint8List(w * (h ~/ 2));

      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: yBuf, rowStride: stride, pixelStride: 1),
        u: (bytes: uvBuf, rowStride: w, pixelStride: 2),
      );

      expect(packed, isNotNull);
      // Row 0 of Y = [0,1,2,3], row 1 = [10,11,12,13]. No stride padding bytes.
      expect(packed!.bytes.sublist(0, w * h), [0, 1, 2, 3, 10, 11, 12, 13]);
    });

    test('tolerates a last row that omits trailing stride padding', () {
      // Real Android delivery: VU buffer length is rows*stride minus
      // the trailing padding of the final row.
      const w = 4, h = 4, stride = 6;
      final yBuf = Uint8List(stride * h);
      final uvRows = h ~/ 2;
      final uvBuf = Uint8List(stride * (uvRows - 1) + w); // short final row

      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: yBuf, rowStride: stride, pixelStride: 1),
        u: (bytes: uvBuf, rowStride: stride, pixelStride: 2),
      );

      expect(packed, isNotNull);
      expect(packed!.bytes.length, w * h + w * uvRows);
    });

    test('rejects odd dimensions', () {
      final p = Uint8List(32);
      expect(
        packYuv420(
          width: 5,
          height: 4,
          y: (bytes: p, rowStride: 5, pixelStride: 1),
          u: (bytes: p, rowStride: 5, pixelStride: 2),
        ),
        isNull,
      );
    });

    test('into buffer produces bytes identical to the allocating form', () {
      const w = 8, h = 4;
      const stride = w + 2;
      final y = Uint8List.fromList(
        List.generate(stride * h, (i) => (i * 7) & 0xff),
      );
      final uv = Uint8List.fromList(
        List.generate(stride * (h ~/ 2), (i) => (i * 13) & 0xff),
      );
      final fresh = packYuv420(
        width: w,
        height: h,
        y: (bytes: y, rowStride: stride, pixelStride: 1),
        u: (bytes: uv, rowStride: stride, pixelStride: 2),
      )!;
      // Pre-fill the reused buffer with garbage: every byte must be written.
      final reuse = Uint8List(w * h * 3 ~/ 2)
        ..fillRange(0, w * h * 3 ~/ 2, 0xAB);
      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: y, rowStride: stride, pixelStride: 1),
        u: (bytes: uv, rowStride: stride, pixelStride: 2),
        into: reuse,
      )!;
      expect(identical(packed.bytes, reuse), isTrue);
      expect(packed.bytes, fresh.bytes);
    });

    test('into buffer zero-fills tails of truncated source planes', () {
      const w = 8, h = 4;
      // Y plane one full row short; the missing row must come back zeroed
      // even when the reused buffer held stale bytes there.
      final y = Uint8List.fromList(List.filled(w * (h - 1), 9));
      final uv = Uint8List.fromList(List.filled(w * (h ~/ 2), 5));
      final reuse = Uint8List(w * h * 3 ~/ 2)
        ..fillRange(0, w * h * 3 ~/ 2, 0xAB);
      final packed = packYuv420(
        width: w,
        height: h,
        y: (bytes: y, rowStride: w, pixelStride: 1),
        u: (bytes: uv, rowStride: w, pixelStride: 2),
        into: reuse,
      )!;
      expect(packed.bytes.sublist(w * (h - 1), w * h), List.filled(w, 0));
    });

    test('into buffer with the wrong length throws', () {
      final p = Uint8List(64);
      expect(
        () => packYuv420(
          width: 4,
          height: 4,
          y: (bytes: p, rowStride: 4, pixelStride: 1),
          u: (bytes: p, rowStride: 4, pixelStride: 2),
          into: Uint8List(10),
        ),
        throwsArgumentError,
      );
    });
  });
}
