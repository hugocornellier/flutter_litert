import 'dart:typed_data';

import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, TargetPlatform;

import 'packed_image_layout.dart';
import 'transferable_bytes.dart';
import 'yuv_conversion.dart';

/// The colour conversion a [CameraFrame]'s bytes need before being used as a
/// 3-channel BGR image. Detector packages map this to an opencv `COLOR_*` code
/// at the point of decode, inside their existing detection isolate.
enum CameraFrameConversion {
  /// 4-channel packed BGRA to 3-channel BGR (macOS camera_desktop).
  bgra2bgr,

  /// 4-channel packed RGBA to 3-channel BGR (Linux camera_desktop).
  rgba2bgr,

  /// YUV420 semi-planar NV12 to BGR (iOS camera plugin default).
  yuv2bgrNv12,

  /// YUV420 semi-planar NV21 to BGR (common Android layout).
  yuv2bgrNv21,

  /// YUV420 planar I420 to BGR (some Android devices).
  yuv2bgrI420,
}

/// Optional rotation applied after colour conversion. Detector packages map
/// this to an opencv `ROTATE_*` code.
enum CameraFrameRotation {
  /// Rotate 90° clockwise.
  cw90,

  /// Rotate 180°.
  cw180,

  /// Rotate 90° counter-clockwise (270° clockwise).
  cw270,
}

/// Operation ordering for a backend-specific [CameraFrame] decoder.
///
/// Four-channel RGB(A) buffers can be resized/rotated before dropping alpha,
/// reducing colour-conversion work. Packed YUV buffers must be colour-converted
/// first because their source layout is not directly resizable as an image.
enum CameraFrameDecodeOrder {
  /// Resize/crop/rotate first, then convert to BGR/RGB/etc.
  resizeRotateThenColorConvert,

  /// Convert colour first, then resize/crop/rotate.
  colorConvertThenResizeRotate,
}

/// Backend-neutral instructions for decoding a [CameraFrame].
///
/// Detector packages map [conversion] and [rotation] to their image backend
/// constants (for example OpenCV `COLOR_*` and `ROTATE_*` values), while this
/// plan centralizes source-buffer layout, stride padding, and operation order.
class CameraFrameDecodePlan {
  /// The packed source layout to reconstruct before colour conversion.
  final PackedImageLayout sourceLayout;

  /// Logical image width before [rotation], excluding stride padding.
  final int visibleWidth;

  /// Logical image height before [rotation].
  final int visibleHeight;

  /// Whether [sourceLayout.cols] contains padded columns beyond [visibleWidth].
  final bool hasStridePadding;

  /// Colour conversion required by the source bytes.
  final CameraFrameConversion conversion;

  /// Optional rotation to apply.
  final CameraFrameRotation? rotation;

  /// Safe operation order for the source layout.
  final CameraFrameDecodeOrder order;

  /// Creates a camera-frame decode plan.
  const CameraFrameDecodePlan({
    required this.sourceLayout,
    required this.visibleWidth,
    required this.visibleHeight,
    required this.hasStridePadding,
    required this.conversion,
    required this.rotation,
    required this.order,
  });
}

/// A camera frame packaged for off-thread colour conversion and inference.
///
/// Produced by [prepareCameraFrame]; consumed by detector packages'
/// `detectFromCameraFrame(...)` methods, which marshal [bytes] across isolate
/// boundaries via `TransferableTypedData` and perform the final `cvtColor` /
/// `rotate` inside their existing detection isolate (off the UI thread).
///
/// The [bytes] buffer is either a tightly-packed YUV420 buffer produced by
/// `packYuv420`, or the raw 4-channel plane of a desktop BGRA/RGBA frame.
///
/// - [width], [height]: the output pixel dimensions before any [rotation].
///   For YUV this matches the luma plane; for BGRA/RGBA this is the logical
///   image size (which may be smaller than the buffer's stride width).
/// - [strideCols]: the Mat column count used when reconstructing a 4-channel
///   Mat from a BGRA/RGBA buffer (`rowStride ~/ 4`). For YUV frames this
///   equals [width] since `packYuv420` produces a tightly-packed buffer.
/// - [conversion]: which colour conversion to apply.
/// - [rotation]: optional post-conversion rotation, or null for none.
class CameraFrame {
  /// The packed pixel bytes, ready for colour conversion.
  final Uint8List bytes;

  /// Output width in pixels (pre-rotation).
  final int width;

  /// Output height in pixels (pre-rotation).
  final int height;

  /// Mat column count when reconstructing a Mat from [bytes]. For BGRA/RGBA
  /// this is `rowStride ~/ 4` and may exceed [width] (in which case the
  /// consumer should crop to [width] × [height]). For YUV this equals [width].
  final int strideCols;

  /// Colour conversion to apply.
  final CameraFrameConversion conversion;

  /// Optional rotation to apply after conversion, or null for none.
  final CameraFrameRotation? rotation;

  const CameraFrame({
    required this.bytes,
    required this.width,
    required this.height,
    required this.strideCols,
    required this.conversion,
    this.rotation,
  });

  /// Builds backend-neutral decode instructions for this frame.
  ///
  /// Image backends should allocate their source buffer from
  /// [CameraFrameDecodePlan.sourceLayout] and use
  /// [PackedImageLayout.copyTo] instead of list-element reconstruction APIs.
  CameraFrameDecodePlan decodePlan() {
    switch (conversion) {
      case CameraFrameConversion.bgra2bgr:
      case CameraFrameConversion.rgba2bgr:
        return CameraFrameDecodePlan(
          sourceLayout: PackedImageLayout(
            rows: height,
            cols: strideCols,
            channels: 4,
            format: conversion == CameraFrameConversion.bgra2bgr
                ? PackedImageFormat.bgra8888
                : PackedImageFormat.rgba8888,
          ),
          visibleWidth: width,
          visibleHeight: height,
          hasStridePadding: strideCols != width,
          conversion: conversion,
          rotation: rotation,
          order: CameraFrameDecodeOrder.resizeRotateThenColorConvert,
        );

      case CameraFrameConversion.yuv2bgrNv12:
      case CameraFrameConversion.yuv2bgrNv21:
      case CameraFrameConversion.yuv2bgrI420:
        final format = switch (conversion) {
          CameraFrameConversion.yuv2bgrNv12 => PackedImageFormat.yuv420Nv12,
          CameraFrameConversion.yuv2bgrNv21 => PackedImageFormat.yuv420Nv21,
          CameraFrameConversion.yuv2bgrI420 => PackedImageFormat.yuv420I420,
          _ => PackedImageFormat.yuv420Nv12,
        };
        return CameraFrameDecodePlan(
          sourceLayout: PackedImageLayout(
            rows: height + height ~/ 2,
            cols: width,
            channels: 1,
            format: format,
          ),
          visibleWidth: width,
          visibleHeight: height,
          hasStridePadding: false,
          conversion: conversion,
          rotation: rotation,
          order: CameraFrameDecodeOrder.colorConvertThenResizeRotate,
        );
    }
  }
}

/// Convenience wrapper around [prepareCameraFrame] that accepts any object
/// duck-typed to `package:camera`'s `CameraImage` (i.e. exposing `width`,
/// `height`, and a `planes` iterable of objects with `bytes`, `bytesPerRow`,
/// and `bytesPerPixel` getters).
///
/// This keeps `flutter_litert` free of a hard dependency on `package:camera`
/// while letting callers pass a `CameraImage` directly:
///
/// ```dart
/// camera.startImageStream((CameraImage image) async {
///   final frame = prepareCameraFrameFromImage(image);
///   if (frame == null) return;
///   final faces = await detector.detectFacesFromCameraFrame(frame);
/// });
/// ```
///
/// Throws at runtime (`NoSuchMethodError` / `TypeError`) if [cameraImage] does
/// not expose the expected shape; this is an acceptable tradeoff vs. either
/// adding a `camera` dep here or asking every caller to write a plane mapper.
///
/// See [prepareCameraFrame] for parameter semantics.
CameraFrame? prepareCameraFrameFromImage(
  Object cameraImage, {
  CameraFrameRotation? rotation,
  bool? isBgra,
}) {
  // ignore: avoid_dynamic_calls
  final dynamic dyn = cameraImage;
  final int width = dyn.width as int;
  final int height = dyn.height as int;
  final Iterable<dynamic> rawPlanes = dyn.planes as Iterable<dynamic>;
  final planes = <CameraPlane>[
    for (final dynamic p in rawPlanes)
      (
        bytes: p.bytes as Uint8List,
        rowStride: p.bytesPerRow as int,
        pixelStride: (p.bytesPerPixel as int?) ?? 1,
      ),
  ];
  return prepareCameraFrame(
    width: width,
    height: height,
    planes: planes,
    rotation: rotation,
    isBgra: isBgra ?? (defaultTargetPlatform == TargetPlatform.macOS),
  );
}

/// Prepare a [CameraFrame] descriptor from raw camera planes, for use with a
/// detector package's `detectFromCameraFrame(...)` method.
///
/// Auto-detects the layout based on plane count and pixel stride:
/// - **1 plane, `pixelStride >= 4`** -> packed BGRA (or RGBA if [isBgra] is
///   false). The `planes` buffer is referenced directly; no copy.
/// - **2 planes** -> NV12. Repacked tightly via `packYuv420`.
/// - **3 planes** -> NV21 or I420 (auto-detected from U pixel stride). Repacked
///   tightly via `packYuv420`.
///
/// Returns null for unsupported shapes (empty planes, missing U plane for
/// YUV, odd width/height for YUV420).
///
/// [isBgra] selects BGRA (macOS, default) vs. RGBA (Linux) for the desktop
/// single-plane path; it is ignored for YUV input.
///
/// Typical usage:
/// ```dart
/// camera.startImageStream((CameraImage image) async {
///   final frame = prepareCameraFrame(
///     width: image.width,
///     height: image.height,
///     planes: [
///       for (final p in image.planes)
///         (bytes: p.bytes, rowStride: p.bytesPerRow,
///          pixelStride: p.bytesPerPixel ?? 1),
///     ],
///   );
///   if (frame == null) return;
///   final faces = await detector.detectFacesFromCameraFrame(frame);
/// });
/// ```
CameraFrame? prepareCameraFrame({
  required int width,
  required int height,
  required List<CameraPlane> planes,
  CameraFrameRotation? rotation,
  bool isBgra = true,
}) {
  if (planes.isEmpty) return null;

  // Desktop single-plane 4-channel BGRA/RGBA.
  if (planes.length == 1 && planes[0].pixelStride >= 4) {
    final p = planes[0];
    return CameraFrame(
      bytes: p.bytes,
      width: width,
      height: height,
      strideCols: p.rowStride ~/ 4,
      conversion: isBgra
          ? CameraFrameConversion.bgra2bgr
          : CameraFrameConversion.rgba2bgr,
      rotation: rotation,
    );
  }

  // YUV420 (2 planes for NV12, 3 for NV21 or I420).
  if (planes.length < 2) return null;
  final y = planes[0];
  final u = planes[1];
  final v = planes.length > 2 ? planes[2] : null;

  final packed = packYuv420(width: width, height: height, y: y, u: u, v: v);
  if (packed == null) return null;

  return CameraFrame(
    bytes: packed.bytes,
    width: packed.width,
    height: packed.height,
    strideCols: packed.width,
    conversion: switch (packed.layout) {
      YuvLayout.nv12 => CameraFrameConversion.yuv2bgrNv12,
      YuvLayout.nv21 => CameraFrameConversion.yuv2bgrNv21,
      YuvLayout.i420 => CameraFrameConversion.yuv2bgrI420,
    },
    rotation: rotation,
  );
}

/// Builds the isolate-request field map for a [CameraFrame] payload, merged with
/// any [extra] per-op fields (e.g. `maxDim`, `mode`, `outputFormat`).
///
/// [frame]'s bytes are wrapped for zero-copy transfer on native (via
/// `TransferableTypedData`) and passed through unchanged on web. Reconstruct on
/// the isolate side with [cameraFrameFromRpcMessage].
Map<String, dynamic> cameraFrameRpcFields(
  CameraFrame frame, [
  Map<String, dynamic> extra = const {},
]) => {
  'bytes': transferableBytes(frame.bytes),
  'width': frame.width,
  'height': frame.height,
  'strideCols': frame.strideCols,
  'conversion': frame.conversion.index,
  'rotation': frame.rotation?.index,
  ...extra,
};

/// Reconstructs a [CameraFrame] from an isolate-request [message] built by
/// [cameraFrameRpcFields] and the already-materialized [bytes].
///
/// The pure-Dart counterpart of each detector's OpenCV decode: the
/// `cameraFrameToBgrMat`-style step that follows stays per-detector (they own
/// the opencv dependency), this only rebuilds the backend-neutral descriptor.
CameraFrame cameraFrameFromRpcMessage(Map message, Uint8List bytes) {
  final int? rotationIndex = message['rotation'] as int?;
  return CameraFrame(
    bytes: bytes,
    width: message['width'] as int,
    height: message['height'] as int,
    strideCols: message['strideCols'] as int,
    conversion: CameraFrameConversion.values[message['conversion'] as int],
    rotation: rotationIndex == null
        ? null
        : CameraFrameRotation.values[rotationIndex],
  );
}
