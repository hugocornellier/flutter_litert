import 'dart:ffi';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// On-device coverage for flutterLitert.runtime=auto.
///
/// Build the host with that property instead of its playServices default:
///
///   env ORG_GRADLE_PROJECT_flutterLitert.runtime=auto flutter test \
///     integration_test/auto_runtime_test.dart \
///     --dart-define=FLUTTER_LITERT_EXPECT_RUNTIME=playServices
///
/// FLUTTER_LITERT_EXPECT_RUNTIME is the runtime the device should end up
/// with: playServices on a device with Google Play services, bundled on one
/// without.
///
/// The tests run in order and share one process: the first one runs before
/// initialization on purpose.
const String _expected = String.fromEnvironment(
  'FLUTTER_LITERT_EXPECT_RUNTIME',
);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List modelBytes;
  String? playServicesError;
  final expected = LiteRtRuntime.values.byName(_expected);

  setUpAll(() async {
    modelBytes = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
  });

  testWidgets('models ask for initialization until a runtime is chosen', (
    tester,
  ) async {
    expect(androidRuntime, isNull);
    final notChosen = isA<StateError>().having(
      (e) => e.message,
      'message',
      allOf(
        contains('flutterLitert.runtime=auto'),
        contains('initializeAndroidRuntime()'),
      ),
    );
    expect(() => Interpreter.fromBuffer(modelBytes), throwsA(notChosen));
    expect(() => CompiledModel.fromBuffer(modelBytes), throwsA(notChosen));
    expect(androidRuntime, isNull);
  });

  testWidgets('initializeAndroidRuntime chooses the expected runtime', (
    tester,
  ) async {
    final status = await initializeAndroidRuntime(
      timeout: const Duration(minutes: 10),
    );
    debugPrint('flutter_litert auto runtime: $status');
    expect(status.runtime, expected);
    expect(status.isReady, isTrue);
    expect(androidRuntime, expected);
    if (expected == LiteRtRuntime.bundled) {
      // Play services itself failed, rather than the plugin around it.
      expect(
        status.playServicesError,
        allOf(
          contains('TfLiteNative.initialize failed'),
          isNot(contains('Android plugin')),
        ),
      );
    } else {
      expect(status.playServicesError, isNull);
    }
    playServicesError = status.playServicesError;
  });

  testWidgets('the APK packages the bundled runtime too', (tester) async {
    expect(() => DynamicLibrary.open('libLiteRt.so'), returnsNormally);
    // LiteRT 1.x is gone: the Interpreter runs on libLiteRt.so too.
    for (final library in [
      'libtensorflowlite_jni.so',
      'libtensorflowlite_gpu_jni.so',
    ]) {
      expect(
        () => DynamicLibrary.open(library),
        throwsArgumentError,
        reason: '$library should not be packaged',
      );
    }
  });

  testWidgets('Interpreter and CompiledModel run', (tester) async {
    final interpreter = Interpreter.fromBuffer(modelBytes);
    addTearDown(interpreter.close);
    interpreter.getInputTensor(0).asFloat32View()[0] = 3;
    interpreter.invoke();
    expect(interpreter.getOutputTensor(0).asFloat32View()[0], 7);

    final model = CompiledModel.fromBuffer(modelBytes);
    addTearDown(model.close);
    final input = Float32List(model.inputByteSizes[0] ~/ 4)..[0] = 3;
    expect(model.run([input])[0][0], 7);
  });

  testWidgets('later calls keep the choice', (tester) async {
    final status = await initializeAndroidRuntime(enableGpu: true);
    expect(status.runtime, expected);
    expect(status.isReady, isTrue);
    expect(status.playServicesError, playServicesError);
  });

  testWidgets('helper isolates use the chosen runtime', (tester) async {
    final bytes = modelBytes;
    final (runtime, output) = await Isolate.run(() {
      final interpreter = Interpreter.fromBuffer(bytes);
      try {
        interpreter.getInputTensor(0).asFloat32View()[0] = 4;
        interpreter.invoke();
        return (
          androidRuntime,
          interpreter.getOutputTensor(0).asFloat32View()[0],
        );
      } finally {
        interpreter.close();
      }
    });
    expect(runtime, expected);
    expect(output, 9);

    // So does the reason for the choice.
    final again = await Isolate.run(
      () async => (await initializeAndroidRuntime()).playServicesError,
    );
    expect(again, playServicesError);
  });
}
