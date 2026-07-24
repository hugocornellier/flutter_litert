import 'dart:io';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

final File _modelFile = File(
  '${Directory.current.path}/test/assets/training_model.tflite',
);

void main() {
  group('collectOutputShapes', () {
    test('returns a shape for every output, keyed by index', () {
      final itp = Interpreter.fromFile(_modelFile);
      try {
        itp.allocateTensors();
        final shapes = collectOutputShapes(itp);
        final expected = itp.getOutputTensors().length;

        expect(shapes, hasLength(expected));
        expect(
          shapes.keys.toList()..sort(),
          List<int>.generate(expected, (i) => i),
        );
        for (final entry in shapes.entries) {
          expect(
            entry.value,
            itp.getOutputTensor(entry.key).shape,
            reason: 'shape mismatch at output ${entry.key}',
          );
          expect(entry.value, isNotEmpty);
        }
      } finally {
        itp.close();
      }
    });

    test('stops at the first missing index rather than throwing', () {
      final itp = Interpreter.fromFile(_modelFile);
      try {
        itp.allocateTensors();
        // The walk is open-ended; it must terminate on its own and never
        // surface the probe exception to the caller.
        expect(() => collectOutputShapes(itp), returnsNormally);
        expect(collectOutputShapes(itp), isNotEmpty);
      } finally {
        itp.close();
      }
    });

    test('is repeatable and returns equal shapes each call', () {
      final itp = Interpreter.fromFile(_modelFile);
      try {
        itp.allocateTensors();
        expect(collectOutputShapes(itp), collectOutputShapes(itp));
      } finally {
        itp.close();
      }
    });
  });
}
