/// A single-slot gate for camera-frame processing that drops frames arriving
/// while a previous frame is still being processed.
///
/// Live camera streams deliver frames faster than on-device inference can keep
/// up. Running every frame would queue work and add latency without improving
/// the result, so the usual pattern is to process a frame only once the
/// previous one has finished and drop the rest. [FrameThrottle] captures that
/// pattern so callers do not have to hand-roll a mutable "busy" flag and the
/// `try`/`finally` that must always clear it (forgetting the `finally` wedges
/// the gate closed forever).
///
/// The busy check and set happen synchronously before the first `await`, so on
/// Dart's single-threaded event loop there is no race between overlapping
/// stream callbacks.
///
/// Usage inside a camera image-stream callback:
/// ```dart
/// final _throttle = FrameThrottle();
///
/// void _onCameraImage(CameraImage image) {
///   _throttle.run(() async {
///     final results = await detector.detectFromCameraImage(image);
///     if (mounted) setState(() => _results = results);
///   });
/// }
/// ```
class FrameThrottle {
  bool _busy = false;

  /// Whether a task is currently in flight.
  bool get isBusy => _busy;

  /// Runs [task] unless another task is already in flight.
  ///
  /// When idle, marks the throttle busy, awaits [task], and returns its result;
  /// the busy flag is always cleared when [task] completes, including when it
  /// throws. When already busy, [task] is not started and this returns `null`
  /// (the frame is dropped).
  Future<T?> run<T>(Future<T> Function() task) async {
    if (_busy) return null;
    _busy = true;
    try {
      return await task();
    } finally {
      _busy = false;
    }
  }

  /// Clears the busy flag without running anything.
  ///
  /// Call when tearing down or switching cameras so a frame that was in flight
  /// does not leave the throttle stuck busy.
  void reset() => _busy = false;
}
