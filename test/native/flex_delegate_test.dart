import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

File get _modelFile =>
    File('${Directory.current.path}/test/assets/training_model.tflite');

void main() {
  group('FlexDelegate', () {
    test('isAvailable reports whether flex library is bundled', () {
      // When running desktop tests, isAvailable depends on whether
      // flutter_litert_flex is present in the project.
      expect(FlexDelegate.isAvailable, isA<bool>());
    });

    test('constructor creates a valid delegate', () {
      final delegate = FlexDelegate();
      expect(delegate.base, isNot(nullptr));
      delegate.delete();
    });

    test('delete throws on double-delete', () {
      final delegate = FlexDelegate();
      delegate.delete();
      expect(() => delegate.delete(), throwsA(isA<StateError>()));
    });
  });

  group('FlexDelegate integration', () {
    late FlexDelegate flexDelegate;
    late InterpreterOptions options;
    late Interpreter interpreter;

    setUp(() {
      flexDelegate = FlexDelegate();
      options = InterpreterOptions();
      options.addDelegate(flexDelegate);
      interpreter = Interpreter.fromFile(_modelFile, options: options);
    });

    tearDown(() {
      interpreter.close();
      flexDelegate.delete();
      options.delete();
    });

    test('interpreter with FlexDelegate can run inference', () {
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
    });

    test('interpreter with FlexDelegate can train and loss decreases', () {
      final train = interpreter.getSignatureRunner('train');
      final loss = Float32List(1);

      // First step: pred=0, target=2, MSE=(0-2)^2=4
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

      // Train for 100 more steps
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
      // After training on y=2x, prediction for x=5 should approach 10
      expect(output[0][0], greaterThan(1.0));
      infer.close();
    });

    test('weight persistence works with FlexDelegate', () {
      // 1. Train
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

      // 2. Get trained weights
      final getW = interpreter.getSignatureRunner('get_weights');
      final w = [
        [0.0],
      ];
      final b = [0.0];
      getW.run({}, {'w': w, 'b': b});
      getW.close();
      expect(w[0][0], isNot(closeTo(0.0, 1e-5)));

      // 3. Record prediction from trained interpreter
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

      // 4. Create FRESH interpreter with FlexDelegate and restore weights
      interpreter.close();
      flexDelegate.delete();
      options.delete();

      flexDelegate = FlexDelegate();
      options = InterpreterOptions();
      options.addDelegate(flexDelegate);
      interpreter = Interpreter.fromFile(_modelFile, options: options);

      // 5. Verify fresh interpreter predicts 0 (untrained)
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

      // 6. Restore weights
      final setW = interpreter.getSignatureRunner('set_weights');
      setW.run({'w': w, 'b': b}, {});
      setW.close();

      // 7. Infer again — should match the trained prediction
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
    });
  });

  // Save/Restore checkpoint integration tests are in
  // test/native/flex_save_restore_test.dart (separate process due to
  // TF runtime atexit crash after tf.raw_ops.Save execution).
}
