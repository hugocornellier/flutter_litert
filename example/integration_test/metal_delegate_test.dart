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
    testWidgets('Metal skipped on non-Apple platform', (_) async {});
    return;
  }

  late File modelFile;

  setUpAll(() async {
    final data = await rootBundle.load('assets/simple_model.tflite');
    final tmpDir = await Directory.systemTemp.createTemp('litert_metal_test_');
    modelFile = File('${tmpDir.path}/simple_model.tflite');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  });

  group('Metal GpuDelegate (macOS)', () {
    testWidgets('GpuDelegate can be created and deleted', (tester) async {
      final delegate = GpuDelegate();
      expect(delegate, isNotNull);
      delegate.delete();
    });

    testWidgets('delete throws on double-delete', (tester) async {
      final delegate = GpuDelegate();
      delegate.delete();
      expect(() => delegate.delete(), throwsA(isA<StateError>()));
    });

    testWidgets('GpuDelegate with options can be created', (tester) async {
      final options = GpuDelegateOptions(
        allowPrecisionLoss: true,
        enableQuantization: false,
      );
      final delegate = GpuDelegate(options: options);
      expect(delegate, isNotNull);
      delegate.delete();
      options.delete();
    });

    testWidgets('inference with Metal delegate produces correct results', (
      tester,
    ) async {
      final delegate = GpuDelegate();
      final opts = InterpreterOptions()..addDelegate(delegate);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      // Interpreter creation retries on CPU when a delegate cannot be applied,
      // so correct numbers alone do not prove Metal ran. Without this the whole
      // suite stays green while the GPU silently degrades to CPU, which is the
      // regression this file exists to catch.
      expect(
        interpreter.hasActiveDelegate,
        isTrue,
        reason: 'Metal delegate did not apply; inference fell back to CPU',
      );

      // Model is y = 2*x + 1
      // f(3) = 7, f(0) = 1, f(-1) = -1
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
      final delegate = GpuDelegate();
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
