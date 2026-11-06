// ignore_for_file: avoid_print

// On-device coverage of the Interpreter features android_model_matrix_test
// cannot reach: signature runners, resize, isolates, the XNNPACK weight cache
// and quantized models. It also samples the custom-op models' raw outputs.
//
// The host tests under test/native cover the same API, but only on the desktop
// runtime. This runs them inside an Android app process.
//
// `>>> FEATURE` lines carry output hashes and sampled values, so two runs (one
// per runtime) can be compared offline. `same-runtime` checks are asserted here.
//
// Models that are not committed are staged by hand into the git-ignored folder
// and the tests that need them skip when they are absent:
//   assets/models/model_matrix/spike_mobilenet_v1_quant.tflite
//   assets/models/model_matrix/spike_yamnet.tflite
//   assets/models/model_matrix/face_detection_tflite__selfie_segmenter*.tflite
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

const _stagedRoot = 'assets/models/model_matrix';
const _sampledValues = 2048;
const _recordChunk = 640;

Future<Uint8List?> _loadStaged(String name) async {
  try {
    return (await rootBundle.load('$_stagedRoot/$name')).buffer.asUint8List();
  } catch (_) {
    return null;
  }
}

/// A float pattern in [0.05, 0.95], the matrix's 'ramp' fixture shape.
Float32List _floatPattern(int values) {
  final result = Float32List(values);
  for (var i = 0; i < values; i++) {
    result[i] = 0.05 + 0.9 * (i % 251) / 250.0;
  }
  return result;
}

Uint8List _bytePattern(int values) {
  final result = Uint8List(values);
  for (var i = 0; i < values; i++) {
    result[i] = (i * 73) % 251;
  }
  return result;
}

void _fillInput(Tensor tensor) {
  if (tensor.type == TensorType.float32) {
    tensor.data = _floatPattern(tensor.numBytes() ~/ 4).buffer.asUint8List();
  } else {
    tensor.data = _bytePattern(tensor.numBytes());
  }
}

/// Emits [bytes] for offline comparison: a SHA-256 and an evenly spaced sample.
void _record(String id, Tensor tensor) {
  final bytes = Uint8List.fromList(tensor.data);
  final isFloat = tensor.type == TensorType.float32;
  final elementSize = isFloat ? 4 : 1;
  final elements = bytes.lengthInBytes ~/ elementSize;
  final stride = max(1, elements ~/ _sampledValues);
  final sample = BytesBuilder();
  for (var i = 0; i < elements; i += stride) {
    sample.add(bytes.sublist(i * elementSize, (i + 1) * elementSize));
  }
  final encoded = base64Encode(sample.toBytes());
  final parts = max(1, (encoded.length / _recordChunk).ceil());
  print(
    '>>> FEATURE kind=output id=$id type=${tensor.type.name} '
    'shape=${tensor.shape.join('x')} sha256=${sha256.convert(bytes)} '
    'stride=$stride',
  );
  for (var part = 0; part < parts; part++) {
    final start = part * _recordChunk;
    final end = min(start + _recordChunk, encoded.length);
    print(
      '>>> FEATURE kind=sample id=$id part=${part + 1}/$parts '
      'data=${encoded.substring(start, end)}',
    );
  }
}

void _value(String id, Object value) {
  print('>>> FEATURE kind=value id=$id value=$value');
}

/// Top-level so the closure sent to the isolate captures only these two values.
Future<Uint8List> _runInIsolate(Uint8List bytes, {required bool xnnpack}) =>
    Isolate.run(() => _runInFreshInterpreter(bytes, xnnpack: xnnpack));

/// Runs [bytes] on one input pattern with a new Interpreter, and returns output
/// 0's bytes.
Uint8List _runInFreshInterpreter(Uint8List bytes, {bool xnnpack = false}) {
  final options = InterpreterOptions()..threads = 2;
  Delegate? delegate;
  if (xnnpack) {
    final xnnOptions = XNNPackDelegateOptions(numThreads: 2);
    delegate = XNNPackDelegate(options: xnnOptions);
    xnnOptions.delete();
    options.addDelegate(delegate);
  }
  final interpreter = Interpreter.fromBuffer(bytes, options: options);
  try {
    interpreter.allocateTensors();
    _fillInput(interpreter.getInputTensor(0));
    interpreter.invoke();
    return Uint8List.fromList(interpreter.getOutputTensor(0).data);
  } finally {
    interpreter.close();
    options.delete();
    delegate?.delete();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List simpleModel;
  late Uint8List trainingModel;
  late Uint8List networkModel;

  setUpAll(() async {
    simpleModel = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
    trainingModel = (await rootBundle.load(
      'assets/training_model.tflite',
    )).buffer.asUint8List();
    networkModel = (await rootBundle.load(
      'assets/mobilefacenet.tflite',
    )).buffer.asUint8List();
  });

  group('signature runners', () {
    late Interpreter interpreter;

    setUp(() => interpreter = Interpreter.fromBuffer(trainingModel));
    tearDown(() => interpreter.close());

    List<List<double>> scalar(double value) => [
      [value],
    ];

    double infer(double x) {
      final runner = interpreter.getSignatureRunner('infer');
      final output = scalar(0);
      runner.run({'x': scalar(x)}, {'output': output});
      runner.close();
      return output[0][0];
    }

    testWidgets('lists all four signatures', (tester) async {
      expect(interpreter.signatureCount, 4);
      expect(
        interpreter.signatureKeys,
        containsAll(<String>['infer', 'train', 'get_weights', 'set_weights']),
      );
    });

    testWidgets('infer starts at zero', (tester) async {
      expect(infer(5.0), closeTo(0.0, 1e-5));
    });

    testWidgets('train reduces the loss from the known first value', (
      tester,
    ) async {
      final runner = interpreter.getSignatureRunner('train');
      addTearDown(runner.close);
      final loss = Float32List(1);
      final losses = <double>[];
      for (var i = 0; i < 20; i++) {
        runner.run({'x': scalar(1.0), 'y': scalar(2.0)}, {'loss': loss});
        losses.add(loss[0]);
      }
      expect(losses.first, closeTo(4.0, 0.01));
      expect(losses.last, lessThan(losses.first));
      _value('signature_losses', losses.join(','));
    });

    testWidgets('set_weights then infer and get_weights', (tester) async {
      final setWeights = interpreter.getSignatureRunner('set_weights');
      setWeights.run({
        'w': scalar(3.0),
        'b': [1.0],
      }, {});
      setWeights.close();
      expect(infer(2.0), closeTo(7.0, 0.01));

      final getWeights = interpreter.getSignatureRunner('get_weights');
      addTearDown(getWeights.close);
      final w = scalar(0);
      final b = Float32List(1);
      getWeights.run({}, {'w': w, 'b': b});
      expect(w[0][0], closeTo(3.0, 1e-5));
      expect(b[0], closeTo(1.0, 1e-5));
    });

    testWidgets('resize on a signature input invalidates handles', (
      tester,
    ) async {
      final runner = interpreter.getSignatureRunner('infer');
      addTearDown(runner.close);
      final before = runner.getInputTensor('x');
      runner.resizeInputTensor('x', [1, 1]);
      runner.allocateTensors();
      final after = runner.getInputTensor('x');
      expect(identical(before, after), isFalse);
      expect(after.shape, [1, 1]);
    });
  });

  group('resize', () {
    /// simple_model is y = 2*x + 1 and broadcasts, so a resized batch has an
    /// exact expected answer for each element.
    void expectBatch(Interpreter interpreter, List<double> xs) {
      final shape = interpreter.getInputTensor(0).shape;
      final newShape = [xs.length, ...shape.skip(1)];
      interpreter.resizeInputTensor(0, newShape);
      interpreter.allocateTensors();
      expect(interpreter.getInputTensor(0).shape, newShape);
      interpreter.getInputTensor(0).data = Float32List.fromList(
        xs,
      ).buffer.asUint8List();
      interpreter.invoke();
      final output = interpreter.getOutputTensor(0).asFloat32View();
      expect(output.length, xs.length);
      for (var i = 0; i < xs.length; i++) {
        expect(output[i], closeTo(2 * xs[i] + 1, 1e-3));
      }
    }

    testWidgets('grows and shrinks a batch on CPU', (tester) async {
      final interpreter = Interpreter.fromBuffer(simpleModel);
      addTearDown(interpreter.close);
      expectBatch(interpreter, [1, 2, 3, 4]);
      expectBatch(interpreter, [5]);
      expectBatch(interpreter, [-1, 0, 1, 2, 3, 4, 5, 6]);
    });

    testWidgets('grows and shrinks a batch on XNNPACK', (tester) async {
      final xnnOptions = XNNPackDelegateOptions(numThreads: 2);
      final delegate = XNNPackDelegate(options: xnnOptions);
      xnnOptions.delete();
      final options = InterpreterOptions()..addDelegate(delegate);
      final interpreter = Interpreter.fromBuffer(simpleModel, options: options);
      addTearDown(() {
        interpreter.close();
        options.delete();
        delegate.delete();
      });
      expectBatch(interpreter, [1, 2, 3, 4]);
      expectBatch(interpreter, [5]);
    });

    testWidgets('resizes a real network to a new spatial size', (tester) async {
      // Output is sampled for the cross-runtime comparison. A model that
      // cannot take the new size is recorded rather than failed: the question
      // is whether both runtimes agree, not whether this model resizes.
      final bytes = (await rootBundle.load(
        'assets/selfie_multiclass.tflite',
      )).buffer.asUint8List();
      final interpreter = Interpreter.fromBuffer(bytes);
      addTearDown(interpreter.close);
      final shape = interpreter.getInputTensor(0).shape;
      try {
        interpreter.resizeInputTensor(0, [shape[0], 320, 320, shape[3]]);
        interpreter.allocateTensors();
        _fillInput(interpreter.getInputTensor(0));
        interpreter.invoke();
        _record('resize_selfie_multiclass_320', interpreter.getOutputTensor(0));
      } catch (error) {
        _value('resize_selfie_multiclass_320_error', error);
      }
    });
  });

  group('isolates', () {
    testWidgets('IsolateInterpreter overlapping calls match a direct run', (
      tester,
    ) async {
      final interpreter = Interpreter.fromBuffer(networkModel);
      addTearDown(interpreter.close);
      interpreter.allocateTensors();
      final inputLength = interpreter.getInputTensor(0).numBytes() ~/ 4;
      final outputLength = interpreter.getOutputTensor(0).numBytes() ~/ 4;

      final inputA = _floatPattern(inputLength);
      final inputB = Float32List(inputLength)..fillRange(0, inputLength, 0.25);

      Float32List direct(Float32List input) {
        final output = Float32List(outputLength);
        interpreter.runForMultipleInputs([input], {0: output});
        return output;
      }

      final expectedA = direct(inputA);
      final expectedB = direct(inputB);

      final isolate = await IsolateInterpreter.create(
        address: interpreter.address,
      );
      addTearDown(isolate.close);
      final outputA = Float32List(outputLength);
      final outputB = Float32List(outputLength);
      await Future.wait([
        isolate.runForMultipleInputs([inputA], {0: outputA}),
        isolate.runForMultipleInputs([inputB], {0: outputB}),
      ]);
      expect(outputA, expectedA);
      expect(outputB, expectedB);
    });

    testWidgets('own Interpreter in four concurrent isolates', (tester) async {
      final expected = _runInFreshInterpreter(networkModel);
      final results = await Future.wait([
        for (var i = 0; i < 4; i++) _runInIsolate(networkModel, xnnpack: false),
      ]);
      for (final result in results) {
        expect(result, expected);
      }
    });

    testWidgets('own XNNPACK Interpreter in four concurrent isolates', (
      tester,
    ) async {
      final expected = _runInFreshInterpreter(networkModel, xnnpack: true);
      final results = await Future.wait([
        for (var i = 0; i < 4; i++) _runInIsolate(networkModel, xnnpack: true),
      ]);
      for (final result in results) {
        expect(result, expected);
      }
    });
  });

  group('XNNPACK weight cache', () {
    testWidgets('writes a cache file and reuses it with the same results', (
      tester,
    ) async {
      final dir = await Directory.systemTemp.createTemp('xnn_weight_cache_');
      addTearDown(() => dir.delete(recursive: true));
      final path = '${dir.path}/mobilefacenet.xnn_cache';

      Uint8List run() {
        final xnnOptions = XNNPackDelegateOptions(
          numThreads: 2,
          weightCacheFilePath: path,
        );
        final delegate = XNNPackDelegate(options: xnnOptions);
        xnnOptions.delete();
        final options = InterpreterOptions()..addDelegate(delegate);
        final interpreter = Interpreter.fromBuffer(
          networkModel,
          options: options,
        );
        try {
          interpreter.allocateTensors();
          _fillInput(interpreter.getInputTensor(0));
          interpreter.invoke();
          return Uint8List.fromList(interpreter.getOutputTensor(0).data);
        } finally {
          interpreter.close();
          options.delete();
          delegate.delete();
        }
      }

      final uncached = _runInFreshInterpreter(networkModel, xnnpack: true);
      final first = run();
      final file = File(path);
      expect(file.existsSync(), isTrue, reason: 'cache file was not written');
      final size = file.lengthSync();
      expect(size, greaterThan(0));
      _value('xnn_cache_bytes', size);
      final second = run();
      expect(second, first);
      expect(first, uncached);
    });
  });

  group('quantized models', () {
    Future<void> sweep(String id, String asset, {required bool xnnpack}) async {
      final bytes = await _loadStaged(asset);
      if (bytes == null) {
        markTestSkipped('$asset is not staged');
        return;
      }
      final options = InterpreterOptions()..threads = 4;
      Delegate? delegate;
      if (xnnpack) {
        final xnnOptions = XNNPackDelegateOptions(numThreads: 4);
        delegate = XNNPackDelegate(options: xnnOptions);
        xnnOptions.delete();
        options.addDelegate(delegate);
      }
      final interpreter = Interpreter.fromBuffer(bytes, options: options);
      addTearDown(() {
        interpreter.close();
        options.delete();
        delegate?.delete();
      });
      interpreter.allocateTensors();
      final input = interpreter.getInputTensor(0);
      _value('${id}_input', '${input.type.name} ${input.params}');
      _fillInput(input);
      interpreter.invoke();
      for (var i = 0; i < interpreter.getOutputTensors().length; i++) {
        _record('${id}_output$i', interpreter.getOutputTensor(i));
      }
    }

    for (final xnnpack in [false, true]) {
      final mode = xnnpack ? 'xnnpack' : 'cpu';
      testWidgets('uint8 mobilenet on $mode', (tester) async {
        await sweep(
          'quant_mobilenet_$mode',
          'spike_mobilenet_v1_quant.tflite',
          xnnpack: xnnpack,
        );
      });
      testWidgets('int8 yamnet on $mode', (tester) async {
        await sweep(
          'quant_yamnet_$mode',
          'spike_yamnet.tflite',
          xnnpack: xnnpack,
        );
      });
    }
  });

  group('custom op models', () {
    for (final model in const [
      'selfie_segmenter',
      'selfie_segmenter_landscape',
    ]) {
      for (final xnnpack in [false, true]) {
        final mode = xnnpack ? 'xnnpack' : 'cpu';
        testWidgets('$model on $mode', (tester) async {
          final bytes = await _loadStaged(
            'face_detection_tflite__$model.tflite',
          );
          if (bytes == null) {
            markTestSkipped('$model is not staged');
            return;
          }
          final options = InterpreterOptions()
            ..threads = 4
            ..addMediaPipeCustomOps();
          Delegate? delegate;
          if (xnnpack) {
            final xnnOptions = XNNPackDelegateOptions(numThreads: 4);
            delegate = XNNPackDelegate(options: xnnOptions);
            xnnOptions.delete();
            options.addDelegate(delegate);
          }
          final interpreter = Interpreter.fromBuffer(bytes, options: options);
          addTearDown(() {
            interpreter.close();
            options.delete();
            delegate?.delete();
          });
          interpreter.allocateTensors();
          _fillInput(interpreter.getInputTensor(0));
          interpreter.invoke();
          final output = interpreter.getOutputTensor(0);
          expect(output.asFloat32View().every((v) => v.isFinite), isTrue);
          _record('custom_op_${model}_$mode', output);
        });
      }
    }
  });
}
