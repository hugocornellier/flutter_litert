// This suite intentionally covers the legacy delegate API until its planned
// removal in flutter_litert 4.0.0.
// ignore_for_file: deprecated_member_use

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  if (!(Platform.isIOS || Platform.isMacOS)) {
    testWidgets('CoreML skipped on non-Apple platform', (_) async {});
    return;
  }

  late File modelFile;

  setUpAll(() async {
    final data = await rootBundle.load('assets/simple_model.tflite');
    final tmpDir = await Directory.systemTemp.createTemp('litert_coreml_test_');
    modelFile = File('${tmpDir.path}/simple_model.tflite');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  });

  group('CoreML Delegate (macOS)', () {
    testWidgets('CoreMlDelegate can be created and deleted', (tester) async {
      final delegate = CoreMlDelegate();
      expect(delegate, isNotNull);
      delegate.delete();
    });

    testWidgets('delete throws on double-delete', (tester) async {
      final delegate = CoreMlDelegate();
      delegate.delete();
      expect(() => delegate.delete(), throwsA(isA<StateError>()));
    });

    testWidgets('CoreMlDelegate with AllDevices option', (tester) async {
      // enabledDevices: 1 = TfLiteCoreMlDelegateAllDevices
      final options = CoreMlDelegateOptions(enabledDevices: 1);
      final delegate = CoreMlDelegate(options: options);
      expect(delegate, isNotNull);
      delegate.delete();
      options.delete();
    });

    testWidgets('inference with CoreML delegate produces correct results', (
      tester,
    ) async {
      // Use AllDevices (1) so it works even without Neural Engine
      final delegate = CoreMlDelegate(
        options: CoreMlDelegateOptions(enabledDevices: 1),
      );
      final opts = InterpreterOptions()..addDelegate(delegate);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      // Model is y = 2*x + 1
      var output = [
        [0.0],
      ];

      interpreter.run([
        [3.0],
      ], output);
      expect(output[0][0], closeTo(7.0, 1e-3));

      interpreter.run([
        [0.0],
      ], output);
      expect(output[0][0], closeTo(1.0, 1e-3));

      interpreter.run([
        [-1.0],
      ], output);
      expect(output[0][0], closeTo(-1.0, 1e-3));

      interpreter.close();
      delegate.delete();
      opts.delete();
    });

    testWidgets('multiple sequential inferences are consistent', (
      tester,
    ) async {
      final delegate = CoreMlDelegate(
        options: CoreMlDelegateOptions(enabledDevices: 1),
      );
      final opts = InterpreterOptions()..addDelegate(delegate);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      final output = [
        [0.0],
      ];
      for (var i = 0; i < 10; i++) {
        interpreter.run([
          [5.0],
        ], output);
        expect(output[0][0], closeTo(11.0, 1e-3));
      }

      interpreter.close();
      delegate.delete();
      opts.delete();
    });
  });
}
