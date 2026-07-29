@TestOn('mac-os')
library;

import 'dart:io';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Guards the ownership fix in [CoreMlDelegate]'s factory.
///
/// When no options are passed, the factory allocates a
/// `TfLiteCoreMlDelegateOptions` and now frees it before returning, instead of
/// leaking it for the life of the process. That is only safe because
/// `TfLiteCoreMlDelegateCreate` copies the struct: TF 2.20.0's `CoreMlDelegate`
/// constructor is `params_(params != nullptr ? *params : ...)`, a copy by value.
///
/// These tests exist to catch the opposite mistake. If the delegate ever starts
/// retaining the pointer, freeing it early becomes a use-after-free, and running
/// real inference through the delegate is what would surface that.
void main() {
  const model = 'test/assets/face_detection_short_range.tflite';

  /// Builds an interpreter through the CoreML path and runs it once.
  ///
  /// Returns false when the CoreML delegate library is not present, in which
  /// case the factory logs a warning and falls back to CPU, so
  /// [CoreMlDelegate] is never constructed and nothing here is exercised.
  /// Reporting that instead of passing anyway matters: a green run that never
  /// touched the code under test is worse than a skip.
  ///
  /// The library ships in the package but is not on the unit-test loader's
  /// search path, so point it there explicitly:
  ///   TFLITE_COREML_PATH=$PWD/macos/flutter_litert/Sources/flutter_litert/\
  ///     Resources/libtensorflowlite_coreml-mac.dylib flutter test
  bool runThroughCoreMl() {
    final (options, delegate) = InterpreterFactory.create(
      PerformanceConfig.coreml(),
    );
    if (delegate == null) return false;
    final interpreter = Interpreter.fromFile(File(model), options: options);
    try {
      interpreter.allocateTensors();
      interpreter.invoke();
      expect(interpreter.getOutputTensors(), isNotEmpty);
      return true;
    } finally {
      interpreter.close();
      delegate.delete();
    }
  }

  test('a delegate created without options survives real inference', () {
    if (!runThroughCoreMl()) {
      markTestSkipped(
        'CoreML delegate library unavailable; set '
        'TFLITE_COREML_PATH to exercise this',
      );
    }
  });

  test('repeated create/destroy cycles stay stable', () {
    // A double free or a use-after-free of the options struct is far likelier
    // to show up across several cycles than on a single one.
    for (var i = 0; i < 5; i++) {
      if (!runThroughCoreMl()) {
        markTestSkipped(
          'CoreML delegate library unavailable; set '
          'TFLITE_COREML_PATH to exercise this',
        );
        return;
      }
    }
  });

  test('caller-supplied options are not freed by the factory', () {
    // The caller owns these, so the factory must leave them alone and they must
    // remain valid for a second delegate afterwards.
    // ignore: deprecated_member_use_from_same_package
    final CoreMlDelegateOptions options;
    // ignore: deprecated_member_use_from_same_package
    final CoreMlDelegate first;
    try {
      // ignore: deprecated_member_use_from_same_package
      options = CoreMlDelegateOptions();
      // ignore: deprecated_member_use_from_same_package
      first = CoreMlDelegate(options: options);
    } on UnsupportedError catch (e) {
      markTestSkipped(
        'CoreML delegate library unavailable ($e); set '
        'TFLITE_COREML_PATH to exercise this',
      );
      return;
    }
    first.delete();

    // Would fail or crash if the factory had already freed the struct.
    // ignore: deprecated_member_use_from_same_package
    final second = CoreMlDelegate(options: options);
    second.delete();

    expect(options.delete, returnsNormally);
  });
}
