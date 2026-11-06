import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

/// Interpreter and CompiledModel sharing the one libLiteRt.so in a bundled
/// Android process, sequentially, concurrently and across isolates.
///
/// interpreter_coexistence_test.dart covers load order and interleaving on the
/// main isolate. Here the two APIs run at the same time on separate threads.
/// Each engine is compared with its own single-threaded reference, because the
/// shared state a problem would corrupt (XNNPACK caches, thread pools, op
/// resolvers) does not show up as a crash reliably.
const int _rounds = 25;

/// One Interpreter setup, sendable to another isolate.
class _Job {
  const _Job(this.model, this.xnnpack, this.customOps);
  final Uint8List model;
  final bool xnnpack;
  final bool customOps;
}

Float32List _pattern(int values) {
  final result = Float32List(values);
  for (var i = 0; i < values; i++) {
    result[i] = 0.05 + 0.9 * (i % 251) / 250.0;
  }
  return result;
}

String _cString(Pointer<Uint8> chars) {
  var length = 0;
  while (chars[length] != 0) {
    length++;
  }
  return utf8.decode(chars.asTypedList(length));
}

/// Runs [rounds] Interpreter invocations and returns each round's output 0.
List<Uint8List> _interpreterLoop(_Job job, int rounds) {
  final options = InterpreterOptions()..threads = 2;
  if (job.customOps) options.addMediaPipeCustomOps();
  Delegate? delegate;
  if (job.xnnpack) {
    final xnnOptions = XNNPackDelegateOptions(numThreads: 2);
    delegate = XNNPackDelegate(options: xnnOptions);
    xnnOptions.delete();
    options.addDelegate(delegate);
  }
  final interpreter = Interpreter.fromBuffer(job.model, options: options);
  try {
    interpreter.allocateTensors();
    final input = interpreter.getInputTensor(0);
    input.data = _pattern(input.numBytes() ~/ 4).buffer.asUint8List();
    final outputs = <Uint8List>[];
    for (var i = 0; i < rounds; i++) {
      interpreter.invoke();
      outputs.add(Uint8List.fromList(interpreter.getOutputTensor(0).data));
    }
    return outputs;
  } finally {
    interpreter.close();
    options.delete();
    delegate?.delete();
  }
}

/// Runs [rounds] CompiledModel CPU invocations and returns each round's output
/// 0.
List<Float32List> _compiledModelLoop(Uint8List model, int rounds) {
  final cm = CompiledModel.fromBuffer(model, accelerators: {Accelerator.cpu});
  try {
    final input = _pattern(cm.inputByteSizes[0] ~/ 4);
    return [
      for (var i = 0; i < rounds; i++) cm.run([input])[0],
    ];
  } finally {
    cm.close();
  }
}

Future<List<Uint8List>> _interpreterInIsolate(_Job job, int rounds) =>
    Isolate.run(() => _interpreterLoop(job, rounds));

Future<List<Float32List>> _compiledModelInIsolate(
  Uint8List model,
  int rounds,
) => Isolate.run(() => _compiledModelLoop(model, rounds));

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List simpleModel;
  late Uint8List networkModel;
  Uint8List? customOpModel;

  setUpAll(() async {
    simpleModel = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
    networkModel = (await rootBundle.load(
      'assets/mobilefacenet.tflite',
    )).buffer.asUint8List();
    try {
      customOpModel = (await rootBundle.load(
        'assets/models/model_matrix/'
        'face_detection_tflite__selfie_segmenter.tflite',
      )).buffer.asUint8List();
    } catch (_) {
      customOpModel = null;
    }
  });

  void expectAllEqual<T>(List<T> actual, T expected, String label) {
    for (var i = 0; i < actual.length; i++) {
      expect(actual[i], expected, reason: '$label, round $i');
    }
  }

  testWidgets('Interpreter runs from libLiteRt.so and no 1.x library exists', (
    tester,
  ) async {
    _interpreterLoop(_Job(simpleModel, false, false), 1);
    _compiledModelLoop(simpleModel, 1);

    // Asks the linker, because /proc/self/maps cannot answer this: libraries
    // are mapped straight from the APK, so every one of them shows as base.apk.
    final version = DynamicLibrary.open('libLiteRt.so')
        .lookupFunction<Pointer<Uint8> Function(), Pointer<Uint8> Function()>(
          'TfLiteVersion',
        );
    expect(_cString(version()), Interpreter.version);
    expect(
      () => DynamicLibrary.open('libtensorflowlite_jni.so'),
      throwsArgumentError,
    );
    expect(
      () => DynamicLibrary.open('libtensorflowlite_gpu_jni.so'),
      throwsArgumentError,
    );
  });

  testWidgets('both APIs on separate isolates at the same time', (
    tester,
  ) async {
    final job = _Job(networkModel, false, false);
    final interpreterExpected = _interpreterLoop(job, 1).single;
    final compiledExpected = _compiledModelLoop(networkModel, 1).single;

    final results = await Future.wait<Object>([
      _interpreterInIsolate(job, _rounds),
      _compiledModelInIsolate(networkModel, _rounds),
      _interpreterInIsolate(job, _rounds),
      _compiledModelInIsolate(networkModel, _rounds),
    ]);

    expectAllEqual(
      results[0] as List<Uint8List>,
      interpreterExpected,
      'interpreter 1',
    );
    expectAllEqual(
      results[1] as List<Float32List>,
      compiledExpected,
      'compiled model 1',
    );
    expectAllEqual(
      results[2] as List<Uint8List>,
      interpreterExpected,
      'interpreter 2',
    );
    expectAllEqual(
      results[3] as List<Float32List>,
      compiledExpected,
      'compiled model 2',
    );
  });

  testWidgets('XNNPACK Interpreter alongside CompiledModel CPU', (
    tester,
  ) async {
    // CompiledModel's CPU path uses XNNPACK too, which is the likeliest place
    // for the two APIs to share state now that they share a library.
    final job = _Job(networkModel, true, false);
    final interpreterExpected = _interpreterLoop(job, 1).single;
    final compiledExpected = _compiledModelLoop(networkModel, 1).single;

    final results = await Future.wait<Object>([
      _interpreterInIsolate(job, _rounds),
      _compiledModelInIsolate(networkModel, _rounds),
    ]);

    expectAllEqual(
      results[0] as List<Uint8List>,
      interpreterExpected,
      'interpreter',
    );
    expectAllEqual(
      results[1] as List<Float32List>,
      compiledExpected,
      'compiled model',
    );
  });

  testWidgets('IsolateInterpreter and CompiledModel.runAsync overlap', (
    tester,
  ) async {
    final interpreter = Interpreter.fromBuffer(networkModel);
    addTearDown(interpreter.close);
    interpreter.allocateTensors();
    final isolate = await IsolateInterpreter.create(
      address: interpreter.address,
    );
    addTearDown(isolate.close);
    final cm = CompiledModel.fromBuffer(
      networkModel,
      accelerators: {Accelerator.cpu},
    );
    addTearDown(cm.close);

    final interpreterInput = _pattern(
      interpreter.getInputTensor(0).numBytes() ~/ 4,
    );
    final outputLength = interpreter.getOutputTensor(0).numBytes() ~/ 4;
    final compiledInput = _pattern(cm.inputByteSizes[0] ~/ 4);

    final interpreterExpected = Float32List(outputLength);
    interpreter.runForMultipleInputs(
      [interpreterInput],
      {0: interpreterExpected},
    );
    final compiledExpected = cm.run([compiledInput])[0];

    for (var i = 0; i < _rounds; i++) {
      final interpreterOutput = Float32List(outputLength);
      final results = await Future.wait<Object?>([
        isolate.runForMultipleInputs(
          [interpreterInput],
          {0: interpreterOutput},
        ),
        cm.runAsync([compiledInput]),
      ]);
      expect(interpreterOutput, interpreterExpected, reason: 'round $i');
      expect(
        (results[1] as List<Float32List>)[0],
        compiledExpected,
        reason: 'round $i',
      );
    }
  });

  testWidgets('custom op Interpreter alongside CompiledModel', (tester) async {
    final model = customOpModel;
    if (model == null) {
      markTestSkipped('selfie_segmenter is not staged');
      return;
    }
    final job = _Job(model, false, true);
    final interpreterExpected = _interpreterLoop(job, 1).single;
    final compiledExpected = _compiledModelLoop(networkModel, 1).single;

    final results = await Future.wait<Object>([
      _interpreterInIsolate(job, _rounds),
      _compiledModelInIsolate(networkModel, _rounds),
    ]);

    expectAllEqual(
      results[0] as List<Uint8List>,
      interpreterExpected,
      'interpreter',
    );
    expectAllEqual(
      results[1] as List<Float32List>,
      compiledExpected,
      'compiled model',
    );
  });

  testWidgets('closing one API while the other runs on another isolate', (
    tester,
  ) async {
    final expected = _compiledModelLoop(networkModel, 1).single;
    final running = _compiledModelInIsolate(networkModel, _rounds);

    for (var i = 0; i < 10; i++) {
      final interpreter = Interpreter.fromBuffer(simpleModel);
      interpreter.allocateTensors();
      interpreter.getInputTensor(0).asFloat32View()[0] = i.toDouble();
      interpreter.invoke();
      expect(
        interpreter.getOutputTensor(0).asFloat32View()[0],
        closeTo(2 * i + 1, 1e-3),
      );
      interpreter.close();
      await tester.pump(Duration.zero);
    }

    expectAllEqual(await running, expected, 'compiled model');
  });

  testWidgets('PerformanceConfig.gpu falls back to XNNPACK', (tester) async {
    final (options, delegate) = InterpreterFactory.create(
      const PerformanceConfig.gpu(),
    );
    final interpreter = Interpreter.fromBuffer(simpleModel, options: options);
    addTearDown(() {
      interpreter.close();
      options.delete();
      delegate?.delete();
    });

    expect(delegate, isA<XNNPackDelegate>());
    interpreter.allocateTensors();
    interpreter.getInputTensor(0).asFloat32View()[0] = 3.0;
    interpreter.invoke();
    expect(
      interpreter.getOutputTensor(0).asFloat32View()[0],
      closeTo(7.0, 1e-3),
    );
  });

  testWidgets('GpuDelegateV2 explains that it is gone on Android', (
    tester,
  ) async {
    expect(() => GpuDelegateV2(), throwsA(isA<UnsupportedError>()));
  });
}
