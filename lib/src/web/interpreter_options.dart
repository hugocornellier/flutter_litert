import 'delegate.dart';

/// Web implementation of InterpreterOptions.
///
/// On web, most options are no-ops. Delegates are accepted but ignored
/// since GPU acceleration is handled by TF.js backend selection.
class InterpreterOptions {
  dynamic get base => null;

  /// Creates a new options instance (no-op container on web).
  factory InterpreterOptions() => InterpreterOptions._();

  InterpreterOptions._();

  /// No-op on web.
  void delete() {}

  /// Ignored on web.
  set threads(int threads) {}

  /// Accepts delegate but ignores it on web.
  void addDelegate(Delegate delegate) {}

  /// No-op on web (custom native ops not available).
  void addMediaPipeCustomOps() {}
}
