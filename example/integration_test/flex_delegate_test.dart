import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late File modelFile;

  setUpAll(() async {
    // Copy the bundled asset to a temp file so the interpreter can read it.
    final data = await rootBundle.load('assets/training_model.tflite');
    final tmpDir = await Directory.systemTemp.createTemp('litert_test_');
    modelFile = File('${tmpDir.path}/training_model.tflite');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  });

  group('FlexDelegate', () {
    testWidgets('isAvailable returns true when flex is bundled', (
      tester,
    ) async {
      expect(FlexDelegate.isAvailable, isTrue);
    });

    testWidgets('create() returns a valid delegate', (tester) async {
      final delegate = await FlexDelegate.create();
      expect(delegate, isNotNull);
      delegate.delete();
    });

    testWidgets('delete throws on double-delete', (tester) async {
      final delegate = await FlexDelegate.create();
      delegate.delete();
      expect(() => delegate.delete(), throwsA(isA<StateError>()));
    });

    testWidgets('inference with FlexDelegate works', (tester) async {
      final flex = await FlexDelegate.create();
      final opts = InterpreterOptions()..addDelegate(flex);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      final infer = interpreter.getSignatureRunner('infer');
      final output = [
        [0.0],
      ];
      infer.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': output},
      );
      // Untrained model, w=0 b=0 → output should be ~0
      expect(output[0][0], closeTo(0.0, 1e-5));
      infer.close();

      interpreter.close();
      flex.delete();
      opts.delete();
    });

    testWidgets('training with FlexDelegate works and loss decreases', (
      tester,
    ) async {
      final flex = await FlexDelegate.create();
      final opts = InterpreterOptions()..addDelegate(flex);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      final train = interpreter.getSignatureRunner('train');
      final loss = Float32List(1);

      // First step
      train.run(
        {
          'x': [
            [1.0],
          ],
          'y': [
            [2.0],
          ],
        },
        {'loss': loss},
      );
      final firstLoss = loss[0];
      expect(firstLoss, closeTo(4.0, 0.01));

      // Train 100 more steps
      for (var i = 0; i < 100; i++) {
        train.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss},
        );
      }
      expect(loss[0], lessThan(firstLoss));
      train.close();

      // Verify prediction improved
      final infer = interpreter.getSignatureRunner('infer');
      final output = [
        [0.0],
      ];
      infer.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': output},
      );
      expect(output[0][0], greaterThan(1.0));
      infer.close();

      interpreter.close();
      flex.delete();
      opts.delete();
    });

    testWidgets('multiple concurrent delegates work independently', (
      tester,
    ) async {
      // Create two delegates simultaneously — exercises the std::map storage
      // on iOS where each delegate is a separate entry.
      final flex1 = await FlexDelegate.create();
      final flex2 = await FlexDelegate.create();

      // Use delegate 1: train for 50 steps
      final opts1 = InterpreterOptions()..addDelegate(flex1);
      final interp1 = Interpreter.fromFile(modelFile, options: opts1);
      final train1 = interp1.getSignatureRunner('train');
      final loss1 = Float32List(1);
      for (var i = 0; i < 50; i++) {
        train1.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss1},
        );
      }
      train1.close();

      // Use delegate 2: untrained inference should give ~0
      final opts2 = InterpreterOptions()..addDelegate(flex2);
      final interp2 = Interpreter.fromFile(modelFile, options: opts2);
      final infer2 = interp2.getSignatureRunner('infer');
      final out2 = [
        [0.0],
      ];
      infer2.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': out2},
      );
      infer2.close();
      expect(out2[0][0], closeTo(0.0, 1e-5));

      // Delegate 1's trained interpreter should give non-zero
      final infer1 = interp1.getSignatureRunner('infer');
      final out1 = [
        [0.0],
      ];
      infer1.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': out1},
      );
      infer1.close();
      expect(out1[0][0], greaterThan(1.0));

      // Clean up both — delete in reverse order to test map erase works
      interp2.close();
      flex2.delete();
      opts2.delete();

      interp1.close();
      flex1.delete();
      opts1.delete();
    });

    testWidgets('delegate reuse across interpreters', (tester) async {
      final flex = await FlexDelegate.create();

      // First interpreter: train
      final opts1 = InterpreterOptions()..addDelegate(flex);
      final interp1 = Interpreter.fromFile(modelFile, options: opts1);
      final train = interp1.getSignatureRunner('train');
      final loss = Float32List(1);
      for (var i = 0; i < 50; i++) {
        train.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss},
        );
      }
      train.close();
      interp1.close();
      opts1.delete();

      // Same delegate, new interpreter: should still work for inference
      final opts2 = InterpreterOptions()..addDelegate(flex);
      final interp2 = Interpreter.fromFile(modelFile, options: opts2);
      final infer = interp2.getSignatureRunner('infer');
      final output = [
        [0.0],
      ];
      infer.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': output},
      );
      infer.close();
      // Fresh interpreter with reused delegate — weights reset, expect ~0
      expect(output[0][0], closeTo(0.0, 1e-5));

      interp2.close();
      flex.delete();
      opts2.delete();
    });

    testWidgets('multiple signature runners on one interpreter', (
      tester,
    ) async {
      final flex = await FlexDelegate.create();
      final opts = InterpreterOptions()..addDelegate(flex);
      final interpreter = Interpreter.fromFile(modelFile, options: opts);

      // 1. Infer on untrained model → ~0
      final infer1 = interpreter.getSignatureRunner('infer');
      final pred1 = [
        [0.0],
      ];
      infer1.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': pred1},
      );
      infer1.close();
      expect(pred1[0][0], closeTo(0.0, 1e-5));

      // 2. Train for 100 steps
      final train = interpreter.getSignatureRunner('train');
      final loss = Float32List(1);
      for (var i = 0; i < 100; i++) {
        train.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss},
        );
      }
      train.close();

      // 3. Infer again → should reflect training
      final infer2 = interpreter.getSignatureRunner('infer');
      final pred2 = [
        [0.0],
      ];
      infer2.run(
        {
          'x': [
            [5.0],
          ],
        },
        {'output': pred2},
      );
      infer2.close();
      expect(pred2[0][0], greaterThan(1.0));

      // 4. Get weights → should be non-zero
      final getW = interpreter.getSignatureRunner('get_weights');
      final w = [
        [0.0],
      ];
      final b = [0.0];
      getW.run({}, {'w': w, 'b': b});
      getW.close();
      expect(w[0][0], isNot(closeTo(0.0, 1e-5)));

      interpreter.close();
      flex.delete();
      opts.delete();
    });

    testWidgets('weight persistence works with FlexDelegate', (tester) async {
      final flex = await FlexDelegate.create();
      final opts = InterpreterOptions()..addDelegate(flex);
      var interpreter = Interpreter.fromFile(modelFile, options: opts);

      // Train
      final train = interpreter.getSignatureRunner('train');
      final loss = Float32List(1);
      for (var i = 0; i < 50; i++) {
        train.run(
          {
            'x': [
              [1.0],
            ],
            'y': [
              [2.0],
            ],
          },
          {'loss': loss},
        );
      }
      train.close();

      // Get trained weights
      final getW = interpreter.getSignatureRunner('get_weights');
      final w = [
        [0.0],
      ];
      final b = [0.0];
      getW.run({}, {'w': w, 'b': b});
      getW.close();
      expect(w[0][0], isNot(closeTo(0.0, 1e-5)));

      // Record trained prediction
      final inferA = interpreter.getSignatureRunner('infer');
      final predA = [
        [0.0],
      ];
      inferA.run(
        {
          'x': [
            [1.0],
          ],
        },
        {'output': predA},
      );
      inferA.close();

      // Fresh interpreter
      interpreter.close();
      flex.delete();
      opts.delete();

      final flex2 = await FlexDelegate.create();
      final opts2 = InterpreterOptions()..addDelegate(flex2);
      interpreter = Interpreter.fromFile(modelFile, options: opts2);

      // Fresh should predict ~0
      final inferFresh = interpreter.getSignatureRunner('infer');
      final predFresh = [
        [0.0],
      ];
      inferFresh.run(
        {
          'x': [
            [1.0],
          ],
        },
        {'output': predFresh},
      );
      inferFresh.close();
      expect(predFresh[0][0], closeTo(0.0, 1e-5));

      // Restore weights
      final setW = interpreter.getSignatureRunner('set_weights');
      setW.run({'w': w, 'b': b}, {});
      setW.close();

      // Should match trained prediction
      final inferB = interpreter.getSignatureRunner('infer');
      final predB = [
        [0.0],
      ];
      inferB.run(
        {
          'x': [
            [1.0],
          ],
        },
        {'output': predB},
      );
      inferB.close();

      expect(predB[0][0], closeTo(predA[0][0], 1e-5));
      expect(predB[0][0], greaterThan(0.5));

      interpreter.close();
      flex2.delete();
      opts2.delete();
    });
  });
}
