import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/native.dart';

File get _modelFile =>
    File('${Directory.current.path}/test/assets/training_model.tflite');

File get _faceModelFile => File(
  '${Directory.current.path}/test/assets/face_detection_short_range.tflite',
);

void main() {
  group('TensorFloat32Views.capture', () {
    late Interpreter interpreter;

    setUp(() {
      interpreter = Interpreter.fromFile(_modelFile);
      interpreter.allocateTensors();
    });

    tearDown(() {
      if (!interpreter.isDeleted) interpreter.close();
    });

    test('returns one view per input tensor', () {
      final views = TensorFloat32Views.capture(interpreter);
      expect(views.inputs.length, equals(interpreter.getInputTensors().length));
      for (final Float32List v in views.inputs) {
        expect(v, isA<Float32List>());
      }
    });

    test('returns one view per output tensor', () {
      final views = TensorFloat32Views.capture(interpreter);
      expect(
        views.outputs.length,
        equals(interpreter.getOutputTensors().length),
      );
      for (final Float32List v in views.outputs) {
        expect(v, isA<Float32List>());
      }
    });

    test('input view length matches tensor byte size / 4', () {
      final views = TensorFloat32Views.capture(interpreter);
      for (int i = 0; i < views.inputs.length; i++) {
        final Tensor t = interpreter.getInputTensor(i);
        expect(
          views.inputs[i].lengthInBytes,
          equals(t.numBytes()),
          reason: 'input $i view bytes should equal tensor byte size',
        );
      }
    });

    test('output view length matches tensor byte size / 4', () {
      final views = TensorFloat32Views.capture(interpreter);
      for (int i = 0; i < views.outputs.length; i++) {
        final Tensor t = interpreter.getOutputTensor(i);
        expect(
          views.outputs[i].lengthInBytes,
          equals(t.numBytes()),
          reason: 'output $i view bytes should equal tensor byte size',
        );
      }
    });

    test('returned view lists are unmodifiable', () {
      final views = TensorFloat32Views.capture(interpreter);
      expect(
        () => views.inputs.add(Float32List(1)),
        throwsA(isA<UnsupportedError>()),
      );
      expect(
        () => views.outputs.add(Float32List(1)),
        throwsA(isA<UnsupportedError>()),
      );
    });
  });

  group('TensorFloat32Views write-through', () {
    late Interpreter interpreter;

    setUp(() {
      interpreter = Interpreter.fromFile(_faceModelFile);
      interpreter.allocateTensors();
    });

    tearDown(() {
      if (!interpreter.isDeleted) interpreter.close();
    });

    /// Reads the float32 element at [index] directly from the input tensor's
    /// native buffer, independently of the view under test.
    double nativeInputFloatAt(int index) {
      final bytes = interpreter.getInputTensor(0).data;
      return ByteData.sublistView(bytes).getFloat32(index * 4, Endian.host);
    }

    test('indexed writes land in the tensor native buffer', () {
      final views = TensorFloat32Views.capture(interpreter);
      views.inputs[0][5] = 3.0;
      views.inputs[0][0] = -1.25;
      expect(nativeInputFloatAt(5), 3.0);
      expect(nativeInputFloatAt(0), -1.25);
    });

    test('per-element fill loop works (natural preprocessing pattern)', () {
      final views = TensorFloat32Views.capture(interpreter);
      final input = views.inputs[0];
      for (var i = 0; i < input.length; i++) {
        input[i] = (i % 7) * 0.5;
      }
      expect(nativeInputFloatAt(0), 0.0);
      expect(nativeInputFloatAt(6), 3.0);
      expect(nativeInputFloatAt(7), 0.0);
    });

    test('setAll writes land in the tensor native buffer', () {
      final views = TensorFloat32Views.capture(interpreter);
      final staged = Float32List(views.inputs[0].length);
      for (var i = 0; i < staged.length; i++) {
        staged[i] = i.toDouble();
      }
      views.inputs[0].setAll(0, staged);
      expect(nativeInputFloatAt(0), 0.0);
      expect(nativeInputFloatAt(1), 1.0);
      expect(nativeInputFloatAt(staged.length - 1), (staged.length - 1));
    });

    test('views-staged inference is byte-identical to setTo-staged', () {
      final rng = math.Random(42);
      final input = Float32List(128 * 128 * 3);
      for (var i = 0; i < input.length; i++) {
        input[i] = rng.nextDouble() * 2 - 1;
      }

      final views = TensorFloat32Views.capture(interpreter);
      views.inputs[0].setAll(0, input);
      interpreter.invoke();
      final viaViews = [
        for (final t in interpreter.getOutputTensors())
          Uint8List.fromList(t.data),
      ];

      final reference = Interpreter.fromFile(_faceModelFile);
      reference.allocateTensors();
      try {
        reference.getInputTensor(0).setTo(input);
        reference.invoke();
        final viaSetTo = [
          for (final t in reference.getOutputTensors())
            Uint8List.fromList(t.data),
        ];
        for (var i = 0; i < viaViews.length; i++) {
          expect(viaViews[i], equals(viaSetTo[i]), reason: 'output $i differs');
        }
      } finally {
        reference.close();
      }
    });
  });
}
