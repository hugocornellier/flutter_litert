import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late File modelFile;

  setUpAll(() async {
    final data = await rootBundle.load('assets/simple_model.tflite');
    final tmpDir = await Directory.systemTemp.createTemp(
      'litert_xnnpack_test_',
    );
    modelFile = File('${tmpDir.path}/simple_model.tflite');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  });

  group('XNNPackDelegate', () {
    testWidgets('XNNPackDelegate can be created and deleted', (tester) async {
      final delegate = XNNPackDelegate();
      expect(delegate, isNotNull);
      delegate.delete();
    });

    testWidgets('delete throws on double-delete', (tester) async {
      final delegate = XNNPackDelegate();
      delegate.delete();
      expect(() => delegate.delete(), throwsA(isA<StateError>()));
    });

    testWidgets('XNNPackDelegate with options can be created', (tester) async {
      final options = XNNPackDelegateOptions(numThreads: 2);
      final delegate = XNNPackDelegate(options: options);
      expect(delegate, isNotNull);
      delegate.delete();
      options.delete();
    });

    testWidgets('inference with XNNPack delegate produces correct results', (
      tester,
    ) async {
      final delegate = XNNPackDelegate();
      final opts = InterpreterOptions()..addDelegate(delegate);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

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
      final delegate = XNNPackDelegate();
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

    testWidgets('auto PerformanceConfig uses XNNPack on non-iOS platforms', (
      tester,
    ) async {
      if (Platform.isIOS) {
        // iOS auto mode uses Metal, not XNNPack.
        return;
      }
      final (options, delegate) = InterpreterFactory.create(
        const PerformanceConfig.auto(),
      );
      expect(delegate, isNotNull);

      final interpreter = Interpreter.fromFile(modelFile, options: options);

      final output = [
        [0.0],
      ];
      interpreter.run([
        [3.0],
      ], output);
      expect(output[0][0], closeTo(7.0, 1e-3));

      interpreter.close();
      delegate!.delete();
      options.delete();
    });
  });
}
