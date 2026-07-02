import 'dart:typed_data';

import 'packed_image_layout.dart';

/// A single YUV plane exposed by a camera plugin, decoupled from any specific
/// Flutter plugin's type (e.g. `CameraImage.Plane`).
///
/// - `bytes`: the plane's raw pixel buffer.
/// - `rowStride`: bytes between the start of consecutive rows (may exceed
///   the logical row width when the buffer is padded).
/// - `pixelStride`: bytes between consecutive pixel samples within a row.
///   1 for planar (I420 U/V). 2 for semi-planar (NV12/NV21 U/V). Always 1
///   for the Y plane.
typedef YuvPlane = ({Uint8List bytes, int rowStride, int pixelStride});

/// A single camera frame plane exposed by a camera plugin.
///
/// Structurally identical to [YuvPlane]. Dart records are compared by shape,
/// so the two names are interchangeable at call sites. Use [CameraPlane] when
/// the plane may be YUV *or* packed BGRA/RGBA (e.g. passing to
/// `cameraFrameToBgrMat`); use [YuvPlane] for the YUV-specific `packYuv420`.
typedef CameraPlane = ({Uint8List bytes, int rowStride, int pixelStride});

/// Memory layout of a packed YUV buffer produced by [packYuv420].
///
/// This enum is intentionally opencv-free: callers wrap [PackedYuv.bytes] in
/// whatever their image library expects. For opencv_dart, map the layout to
/// the matching `COLOR_YUV2BGR_*` / `COLOR_YUV2RGB_*` conversion code.
enum YuvLayout {
  /// Y plane followed by interleaved U,V,U,V,... (iOS camera default).
  nv12,

  /// Y plane followed by interleaved V,U,V,U,... (most Android devices).
  nv21,

  /// Y plane, then full U plane, then full V plane (planar Android).
  i420,
}

/// A contiguous YUV buffer produced by [packYuv420], ready to hand to a
/// native colour-conversion routine.
///
/// The [bytes] buffer has length `width * height * 3 ~/ 2` and is laid out as
/// described by [layout]. [width] and [height] refer to the luma (Y) plane.
class PackedYuv {
  /// Packed YUV bytes (Y plane first, chroma layout per [layout]).
  final Uint8List bytes;

  /// Memory layout of [bytes].
  final YuvLayout layout;

  /// Luma plane width, in pixels.
  final int width;

  /// Luma plane height, in pixels.
  final int height;

  const PackedYuv({
    required this.bytes,
    required this.layout,
    required this.width,
    required this.height,
  });

  /// Source layout for reconstructing this packed YUV buffer in an image
  /// backend.
  PackedImageLayout get sourceLayout => PackedImageLayout(
    rows: height + height ~/ 2,
    cols: width,
    channels: 1,
    format: switch (layout) {
      YuvLayout.nv12 => PackedImageFormat.yuv420Nv12,
      YuvLayout.nv21 => PackedImageFormat.yuv420Nv21,
      YuvLayout.i420 => PackedImageFormat.yuv420I420,
    },
  );
}

/// Packs a YUV420 camera frame into a single contiguous buffer suitable for
/// native colour conversion (e.g. opencv's `cvtColor` with a
/// `COLOR_YUV2BGR_NV21` / `COLOR_YUV2BGR_NV12` / `COLOR_YUV2BGR_I420` code).
///
/// Auto-detects the source layout based on the plane count and the U plane's
/// `YuvPlane.pixelStride`:
///
/// - **2 planes: NV12.** iOS `AVFoundation` default.
/// - **3 planes, U pixelStride 2: NV21.** Most Android devices (semi-planar);
///   the V plane's buffer is used as the VU-interleaved region start.
/// - **3 planes, U pixelStride 1: I420.** Planar Android.
///
/// Row-stride padding is stripped so the returned buffer is tightly packed.
///
/// Typical usage with opencv_dart:
/// ```dart
/// final packed = packYuv420(
///   width: image.width,
///   height: image.height,
///   y: (bytes: image.planes[0].bytes,
///       rowStride: image.planes[0].bytesPerRow,
///       pixelStride: image.planes[0].bytesPerPixel ?? 1),
///   u: (bytes: image.planes[1].bytes,
///       rowStride: image.planes[1].bytesPerRow,
///       pixelStride: image.planes[1].bytesPerPixel ?? 1),
///   v: image.planes.length > 2
///     ? (bytes: image.planes[2].bytes,
///        rowStride: image.planes[2].bytesPerRow,
///        pixelStride: image.planes[2].bytesPerPixel ?? 1)
///     : null,
/// );
///
/// final code = switch (packed.layout) {
///   YuvLayout.nv12 => cv.COLOR_YUV2BGR_NV12,
///   YuvLayout.nv21 => cv.COLOR_YUV2BGR_NV21,
///   YuvLayout.i420 => cv.COLOR_YUV2BGR_I420,
/// };
/// final layout = packed.sourceLayout;
/// final yuvMat = cv.Mat.create(
///   rows: layout.rows,
///   cols: layout.cols,
///   type: cv.MatType.CV_8UC1,
/// );
/// layout.copyTo(yuvMat.data, packed.bytes);
/// final bgr = cv.cvtColor(yuvMat, code);
/// yuvMat.dispose();
/// ```
///
/// Returns null for unsupported shapes (non-positive or odd [width] or
/// [height]).
///
/// [into] optionally receives the packed bytes instead of allocating a
/// fresh buffer, which matters in per-frame camera loops: a 720p pack
/// otherwise allocates and zero-fills ~1.4 MB per call. Its length must be
/// exactly `width * height * 3 ~/ 2` (throws [ArgumentError] otherwise).
/// Every byte is written (truncated source planes zero-fill their tail),
/// so a reused buffer needs no clearing and the output is byte-identical
/// to the allocating form.
PackedYuv? packYuv420({
  required int width,
  required int height,
  required YuvPlane y,
  required YuvPlane u,
  YuvPlane? v,
  Uint8List? into,
}) {
  if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0) {
    return null;
  }

  final int ySize = width * height;
  final int uvSize = width * (height ~/ 2);
  if (into != null && into.length != ySize + uvSize) {
    throw ArgumentError.value(
      into.length,
      'into.length',
      'Expected ${ySize + uvSize} bytes for ${width}x$height YUV420.',
    );
  }
  final Uint8List out = into ?? Uint8List(ySize + uvSize);

  _copyPlaneRows(
    src: y.bytes,
    srcStride: y.rowStride,
    rowBytes: width,
    rows: height,
    dst: out,
    dstOffset: 0,
  );

  if (v == null) {
    // 2-plane NV12: planes[1] is the UV-interleaved chroma region.
    _copyPlaneRows(
      src: u.bytes,
      srcStride: u.rowStride,
      rowBytes: width,
      rows: height ~/ 2,
      dst: out,
      dstOffset: ySize,
    );
    return PackedYuv(
      bytes: out,
      layout: YuvLayout.nv12,
      width: width,
      height: height,
    );
  }

  if (u.pixelStride == 2) {
    // Android semi-planar (NV21). plane[2] points to the V byte, which is
    // the first byte of the VU-interleaved region when the device uses NV21.
    _copyPlaneRows(
      src: v.bytes,
      srcStride: v.rowStride,
      rowBytes: width,
      rows: height ~/ 2,
      dst: out,
      dstOffset: ySize,
    );
    return PackedYuv(
      bytes: out,
      layout: YuvLayout.nv21,
      width: width,
      height: height,
    );
  }

  // Planar I420.
  final int uvWidth = width ~/ 2;
  final int uvHeight = height ~/ 2;
  _copyPlaneRows(
    src: u.bytes,
    srcStride: u.rowStride,
    rowBytes: uvWidth,
    rows: uvHeight,
    dst: out,
    dstOffset: ySize,
  );
  _copyPlaneRows(
    src: v.bytes,
    srcStride: v.rowStride,
    rowBytes: uvWidth,
    rows: uvHeight,
    dst: out,
    dstOffset: ySize + uvWidth * uvHeight,
  );
  return PackedYuv(
    bytes: out,
    layout: YuvLayout.i420,
    width: width,
    height: height,
  );
}

void _copyPlaneRows({
  required Uint8List src,
  required int srcStride,
  required int rowBytes,
  required int rows,
  required Uint8List dst,
  required int dstOffset,
}) {
  // Truncated sources zero-fill the uncopied tail so output is identical
  // whether dst is freshly allocated (already zero) or a reused buffer.
  if (srcStride == rowBytes) {
    final int total = rowBytes * rows;
    final int copy = total <= src.length ? total : src.length;
    dst.setRange(dstOffset, dstOffset + copy, src);
    if (copy < total) {
      dst.fillRange(dstOffset + copy, dstOffset + total, 0);
    }
    return;
  }
  for (int r = 0; r < rows; r++) {
    final int sStart = r * srcStride;
    final int dStart = dstOffset + r * rowBytes;
    if (sStart >= src.length) {
      dst.fillRange(dStart, dstOffset + rows * rowBytes, 0);
      return;
    }
    final int available = src.length - sStart;
    final int copy = available < rowBytes ? available : rowBytes;
    dst.setRange(dStart, dStart + copy, src, sStart);
    if (copy < rowBytes) {
      dst.fillRange(dStart + copy, dStart + rowBytes, 0);
    }
  }
}
