import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// flutterLitert.runtime=playServices on a device without Google Play
/// services, such as an emulator with a plain `default` system image.
///
/// Initialization must fail promptly rather than hang, explain both failures,
/// and leave models failing with that explanation. A retry fails the same way.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List modelBytes;

  setUpAll(() async {
    modelBytes = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
  });

  Future<LiteRtRuntimeUnavailableException> expectUnavailable() async {
    try {
      await initializeAndroidRuntime(
        enableGpu: true,
      ).timeout(const Duration(minutes: 2));
    } on LiteRtRuntimeUnavailableException catch (e) {
      return e;
    }
    fail('initializeAndroidRuntime() succeeded without Google Play services');
  }

  testWidgets('initializeAndroidRuntime throws with both reasons', (
    tester,
  ) async {
    expect(androidRuntime, LiteRtRuntime.playServices);
    final error = await expectUnavailable();
    debugPrint('flutter_litert without Play services: $error');

    final status = error.status;
    expect(status.runtime, LiteRtRuntime.playServices);
    expect(status.isReady, isFalse);
    expect(status.interpreterReady, isFalse);
    expect(status.compiledModelReady, isFalse);
    expect(status.interpreterError, isNotNull);
    expect(status.compiledModelError, isNotNull);
  });

  testWidgets('models report the failure and how to retry', (tester) async {
    final unavailable = isA<StateError>().having(
      (e) => e.message,
      'message',
      allOf(contains('unavailable'), contains('initializeAndroidRuntime()')),
    );
    expect(() => Interpreter.fromBuffer(modelBytes), throwsA(unavailable));
    expect(() => CompiledModel.fromBuffer(modelBytes), throwsA(unavailable));
  });

  testWidgets('a retry fails the same way', (tester) async {
    final error = await expectUnavailable();
    expect(error.status.isReady, isFalse);
  });
}
