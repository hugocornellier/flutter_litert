// Unsupported platform stub for InterpreterOptions.
//
// This file is used when compiling for platforms where neither dart:io
// nor dart:js_interop is available.

import 'delegate.dart';

/// TensorFlowLite interpreter options.
class InterpreterOptions {
  dynamic get base => throw UnsupportedError(
    'InterpreterOptions.base is not supported on this platform',
  );

  /// Creates a new options instance.
  factory InterpreterOptions() => throw UnsupportedError(
    'InterpreterOptions is not supported on this platform',
  );

  /// Destroys the options instance.
  void delete() => throw UnsupportedError(
    'InterpreterOptions.delete is not supported on this platform',
  );

  /// Sets the number of CPU threads to use.
  set threads(int threads) => throw UnsupportedError(
    'InterpreterOptions.threads is not supported on this platform',
  );

  /// Adds delegate to Interpreter Options
  void addDelegate(Delegate delegate) => throw UnsupportedError(
    'InterpreterOptions.addDelegate is not supported on this platform',
  );

  /// Registers MediaPipe custom ops (like Convolution2DTransposeBias).
  void addMediaPipeCustomOps() => throw UnsupportedError(
    'InterpreterOptions.addMediaPipeCustomOps is not supported on this platform',
  );
}
