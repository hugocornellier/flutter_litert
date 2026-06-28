import 'delegate.dart';

/// Web implementation of InterpreterOptions.
///
/// On web, most options are no-ops. Delegates are accepted for API
/// compatibility but ignored by the tflite-js `Interpreter`. Use
/// `LiteRtInterpreter` for LiteRT.js WebGPU execution.
class InterpreterOptions {
  dynamic get base => null;

  int? _threads;
  bool _hasDelegate = false;

  /// Creates a new options instance (no-op container on web).
  factory InterpreterOptions() => InterpreterOptions._();

  InterpreterOptions._();

  /// Whether a delegate was added. Mirrors the native API; delegates are
  /// accepted for compatibility but ignored on web.
  bool get hasDelegate => _hasDelegate;

  /// The configured thread count, if any.
  int? get threads => _threads;

  /// No-op on web.
  void delete() {}

  /// Recorded for API parity but ignored by the tflite-js interpreter.
  set threads(int threads) => _threads = threads;

  /// Accepts delegate but ignores it on web.
  void addDelegate(Delegate delegate) => _hasDelegate = true;

  /// No-op on web (custom native ops not available).
  void addMediaPipeCustomOps() {}

  /// Mirrors the native fallback helper: a copy with the same thread tuning but
  /// no delegates.
  InterpreterOptions copyWithoutDelegates() =>
      InterpreterOptions().._threads = _threads;
}
