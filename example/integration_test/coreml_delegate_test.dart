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
  late File meanPoolingModelFile;

  setUpAll(() async {
    final data = await rootBundle.load('assets/simple_model.tflite');
    final meanPoolingData = await rootBundle.load(
      'assets/species_classifier_float16.tflite',
    );
    final tmpDir = await Directory.systemTemp.createTemp('litert_coreml_test_');
    modelFile = File('${tmpDir.path}/simple_model.tflite');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
    meanPoolingModelFile = File(
      '${tmpDir.path}/species_classifier_float16.tflite',
    );
    await meanPoolingModelFile.writeAsBytes(
      meanPoolingData.buffer.asUint8List(),
    );
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

      // Interpreter creation retries on CPU when a delegate cannot be applied,
      // so correct numbers alone do not prove CoreML ran. This holds only
      // because the delegate above requests AllDevices: the default
      // (DevicesWithNeuralEngine) does not apply on a Mac or a simulator, and
      // would legitimately leave hasActiveDelegate false.
      expect(
        interpreter.hasActiveDelegate,
        isTrue,
        reason: 'CoreML delegate did not apply; inference fell back to CPU',
      );

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

    if (Platform.isMacOS) {
      testWidgets('global MEAN model applies without falling back to CPU', (
        tester,
      ) async {
        // This model reaches PoolingLayerBuilder's global-MEAN path. A
        // stock TensorFlow 2.20 CoreML delegate leaves its required padding
        // oneof unset, fails Core ML compilation, and silently retries on
        // CPU. The packaged dylib carries coreml_mean_padding.patch.
        final delegateOptions = CoreMlDelegateOptions(enabledDevices: 1);
        final delegate = CoreMlDelegate(options: delegateOptions);
        final options = InterpreterOptions()..addDelegate(delegate);
        final interpreter = Interpreter.fromFile(
          meanPoolingModelFile,
          options: options,
        );

        expect(
          interpreter.hasActiveDelegate,
          isTrue,
          reason:
              'The global-MEAN CoreML model did not compile; the packaged '
              'delegate may be missing coreml_mean_padding.patch',
        );

        interpreter.close();
        delegate.delete();
        options.delete();
        delegateOptions.delete();
      });
    }

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
