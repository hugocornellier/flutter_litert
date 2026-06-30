import 'dart:ui' show Size;

import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform;
import 'package:flutter/services.dart' show DeviceOrientation;

import 'camera_frame.dart' show CameraFrameRotation;

/// Compute the rotation needed to present a camera frame upright to an
/// on-device detection model, given the camera's sensor orientation and the
/// device's current physical orientation.
///
/// - **iOS**: assumes the camera plugin pre-rotates the image stream per
///   `AVCaptureConnection.videoOrientation` (portrait-only path). Returns a
///   rotation only when the device is in portrait and the frame arrived in
///   landscape-sensor layout.
/// - **Android**: combined `(sensor ± deviceRotation) % 360` formula; the
///   sign depends on front vs. back camera.
/// - **Other platforms** (desktop / web): returns null; `camera_desktop` and
///   the web backend deliver already-upright frames.
///
/// Callers typically pass `image.width` / `image.height` from a
/// `CameraImage`, the camera's `sensorOrientation` (via
/// `CameraDescription.sensorOrientation`), and the effective device
/// orientation (via `CameraController.value.deviceOrientation` on mobile, or
/// a fallback based on `MediaQuery` when the controller is still
/// initializing).
CameraFrameRotation? rotationForFrame({
  required int width,
  required int height,
  required int sensorOrientation,
  required bool isFrontCamera,
  required DeviceOrientation deviceOrientation,
}) {
  if (defaultTargetPlatform == TargetPlatform.iOS) {
    final bool isPortrait =
        deviceOrientation == DeviceOrientation.portraitUp ||
        deviceOrientation == DeviceOrientation.portraitDown;
    if (!isPortrait) return null;
    if (height >= width) return null;
    if (sensorOrientation == 90) return CameraFrameRotation.cw90;
    if (sensorOrientation == 270) return CameraFrameRotation.cw270;
    return null;
  }

  if (defaultTargetPlatform == TargetPlatform.android) {
    final int deviceRotation = switch (deviceOrientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
    };

    final int total = isFrontCamera
        ? (sensorOrientation + deviceRotation) % 360
        : (sensorOrientation - deviceRotation + 360) % 360;

    return switch (total) {
      90 => CameraFrameRotation.cw90,
      180 => CameraFrameRotation.cw180,
      270 => CameraFrameRotation.cw270,
      _ => null,
    };
  }

  return null;
}

/// Compute the final detection-image size used by overlay painters to map
/// detector coordinates back onto the widget coord space.
///
/// Deterministic from the same inputs the detection isolate receives: the
/// pre-rotation [width] / [height], the optional [rotation] (swaps dims when
/// 90/270), and an optional [maxDim] downscale (preserves aspect ratio).
///
/// Returns the post-rotation, post-downscale size in pixels. Pass this as the
/// source size of your overlay painter's coordinate mapping.
Size detectionSize({
  required int width,
  required int height,
  required CameraFrameRotation? rotation,
  required int maxDim,
}) {
  int w = width;
  int h = height;
  if (rotation == CameraFrameRotation.cw90 ||
      rotation == CameraFrameRotation.cw270) {
    final int t = w;
    w = h;
    h = t;
  }
  if (w > maxDim || h > maxDim) {
    final double scale = maxDim / (w > h ? w : h);
    w = (w * scale).toInt();
    h = (h * scale).toInt();
  }
  return Size(w.toDouble(), h.toDouble());
}

/// Cover-fit scale + offset for rendering a source region of size
/// ([sourceW], [sourceH]) into a viewport of size ([viewW], [viewH]).
///
/// Preserves aspect ratio and centers; the source is scaled to cover the
/// viewport (by the larger of the width/height ratios), so the axis that
/// overflows is centered with a zero or negative offset and the other axis
/// offset is zero. The record `(scale, offsetX, offsetY)` is what a
/// `CustomPainter` typically needs to transform source coordinates to
/// viewport coordinates: `x_view = x_source * scale + offsetX`.
({double scale, double offsetX, double offsetY}) coverFitScaleOffset(
  int sourceW,
  int sourceH,
  double viewW,
  double viewH,
) {
  final double sourceAspect = sourceW / sourceH;
  final double viewAspect = viewW / viewH;
  if (sourceAspect > viewAspect) {
    final double s = viewH / sourceH;
    return (scale: s, offsetX: (viewW - sourceW * s) / 2, offsetY: 0.0);
  }
  final double s = viewW / sourceW;
  return (scale: s, offsetX: 0.0, offsetY: (viewH - sourceH * s) / 2);
}

/// Quarter-turns (clockwise) to rotate a top-bar widget so it reads upright
/// when the device is in landscape. Use with `RotatedBox(quarterTurns: ...)`.
///
/// Returns 0 for portrait (up or down), 1 for landscape-left, 3 for
/// landscape-right.
int barQuarterTurns(DeviceOrientation orientation) {
  return switch (orientation) {
    DeviceOrientation.landscapeLeft => 1,
    DeviceOrientation.landscapeRight => 3,
    _ => 0,
  };
}

/// A simple 1-second rolling FPS counter for camera-preview apps.
///
/// Call [tick] once per processed frame; [tick] returns `true` at most once
/// per second (when [fps] has been refreshed), so the caller can use its
/// return value to decide whether to trigger a widget rebuild.
///
/// Usage:
/// ```dart
/// final _fpsCounter = FpsCounter();
/// int _fps = 0;
///
/// void onFrame() {
///   if (_fpsCounter.tick() && mounted) {
///     setState(() => _fps = _fpsCounter.fps);
///   }
/// }
/// ```
class FpsCounter {
  int _fps = 0;
  int _framesSinceLastUpdate = 0;
  DateTime? _lastUpdate;

  /// The most recently computed FPS value. Starts at 0 until the first
  /// 1-second interval completes.
  int get fps => _fps;

  /// Record a frame. Returns `true` when [fps] was refreshed (i.e. at most
  /// once per second); callers typically guard `setState(...)` on the result.
  bool tick() {
    _framesSinceLastUpdate++;
    final now = DateTime.now();
    if (_lastUpdate == null) {
      _lastUpdate = now;
      return false;
    }
    final int diff = now.difference(_lastUpdate!).inMilliseconds;
    if (diff >= 1000) {
      _fps = (_framesSinceLastUpdate * 1000 / diff).round();
      _framesSinceLastUpdate = 0;
      _lastUpdate = now;
      return true;
    }
    return false;
  }

  /// Reset all counters. Useful when switching cameras.
  void reset() {
    _fps = 0;
    _framesSinceLastUpdate = 0;
    _lastUpdate = null;
  }
}
