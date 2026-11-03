import 'dart:ffi';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'real_models.dart';

/// On-device coverage for flutterLitert.runtime=playServices.
///
/// This host app is built with that property (android/gradle.properties), so
/// LiteRT comes from Google Play services instead of the APK. The device needs
/// Google Play services; the runtime module is downloaded on first use.
///
/// The tests run in order and share one process: the first group runs before
/// initialization on purpose.
///
/// simple_model is y = 2*x + 1, so results are exact. The real-model group
/// (real_models.dart) checks two real networks against reference outputs.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List modelBytes;
  late LiteRtRuntimeStatus status;

  setUpAll(() async {
    modelBytes = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
  });

  double runInterpreter(Interpreter interpreter, double x) {
    interpreter.getInputTensor(0).asFloat32View()[0] = x;
    interpreter.invoke();
    return interpreter.getOutputTensor(0).asFloat32View()[0];
  }

  double runCompiledModel(CompiledModel model, double x) {
    final input = Float32List(model.inputByteSizes[0] ~/ 4);
    input[0] = x;
    return model.run([input])[0][0];
  }

  group('before initializeAndroidRuntime', () {
    testWidgets('the app was built for Play services', (tester) async {
      expect(androidRuntime, LiteRtRuntime.playServices);
    });

    testWidgets('the APK does not bundle LiteRT', (tester) async {
      for (final library in [
        'libLiteRt.so',
        'libtensorflowlite_jni.so',
        'libtensorflowlite_gpu_jni.so',
        'libLiteRtClGlAccelerator.so',
      ]) {
        expect(
          () => DynamicLibrary.open(library),
          throwsArgumentError,
          reason: '$library should not be packaged',
        );
      }
    });

    testWidgets('Interpreter explains that initialization is required', (
      tester,
    ) async {
      expect(
        () => Interpreter.fromBuffer(modelBytes),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('initializeAndroidRuntime()'),
          ),
        ),
      );
    });

    testWidgets('CompiledModel explains that initialization is required', (
      tester,
    ) async {
      expect(
        () => CompiledModel.fromBuffer(modelBytes),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('initializeAndroidRuntime()'),
          ),
        ),
      );
    });
  });

  group('initializeAndroidRuntime', () {
    testWidgets('initializes both runtimes', (tester) async {
      // Concurrent callers share one initialization.
      final results = await Future.wait([
        initializeAndroidRuntime(enableGpu: true),
        initializeAndroidRuntime(enableGpu: true),
      ]);
      status = results[0];
      debugPrint('flutter_litert Play services status: $status');

      expect(status.runtime, LiteRtRuntime.playServices);
      expect(status.interpreterError, isNull);
      expect(status.compiledModelError, isNull);
      expect(status.interpreterReady, isTrue);
      expect(status.compiledModelReady, isTrue);
      expect(status.isReady, isTrue);
      expect(status.gpuAvailable, isNotNull);
      if (status.gpuAvailable == false) {
        expect(
          status.gpuError,
          isNotNull,
          reason: 'a missing GPU is explained',
        );
      }
      expect(results[1].isReady, isTrue);
    });

    testWidgets('returns at once when already initialized', (tester) async {
      final again = await initializeAndroidRuntime();
      expect(again.isReady, isTrue);
      expect(again.gpuAvailable, status.gpuAvailable);
      expect(again.gpuError, status.gpuError);
    });
  });

  group('Interpreter', () {
    testWidgets('runs on the Play services runtime', (tester) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      addTearDown(interpreter.close);

      expect(runInterpreter(interpreter, 3.0), closeTo(7.0, 1e-3));
      expect(runInterpreter(interpreter, -1.0), closeTo(-1.0, 1e-3));
    });

    testWidgets('runs in a helper isolate without initializing again', (
      tester,
    ) async {
      final bytes = modelBytes;
      final y = await Isolate.run(() {
        final interpreter = Interpreter.fromBuffer(bytes);
        try {
          return runInterpreter(interpreter, 5.0);
        } finally {
          interpreter.close();
        }
      });
      expect(y, closeTo(11.0, 1e-3));
    });

    testWidgets('runs through IsolateInterpreter', (tester) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      addTearDown(interpreter.close);
      final isolate = await IsolateInterpreter.create(
        address: interpreter.address,
      );
      addTearDown(isolate.close);

      final input = [
        [2.0],
      ];
      final output = [
        [0.0],
      ];
      await isolate.run(input, output);
      expect(output[0][0], closeTo(5.0, 1e-3));
    });

    testWidgets('PerformanceMode.auto uses no delegate and runs', (
      tester,
    ) async {
      final (options, delegate) = InterpreterFactory.create(
        const PerformanceConfig(mode: PerformanceMode.auto),
      );
      expect(delegate, isNull);
      final interpreter = Interpreter.fromBuffer(modelBytes, options: options);
      addTearDown(interpreter.close);
      expect(runInterpreter(interpreter, 1.0), closeTo(3.0, 1e-3));
    });

    testWidgets('features Play services lacks throw UnsupportedError', (
      tester,
    ) async {
      final playServicesError = isA<UnsupportedError>().having(
        (e) => e.message,
        'message',
        contains('Google Play services'),
      );
      expect(() => XNNPackDelegate(), throwsA(playServicesError));
      expect(() => GpuDelegateV2(), throwsA(playServicesError));
      final options = InterpreterOptions();
      addTearDown(options.delete);
      expect(options.addMediaPipeCustomOps, throwsA(playServicesError));
    });
  });

  group('CompiledModel', () {
    testWidgets('runs on CPU', (tester) async {
      final model = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(model.close);

      expect(runCompiledModel(model, 3.0), closeTo(7.0, 1e-3));
      expect(runCompiledModel(model, 0.0), closeTo(1.0, 1e-3));
    });

    testWidgets('runs in a helper isolate without initializing again', (
      tester,
    ) async {
      final bytes = modelBytes;
      final y = await Isolate.run(() {
        final model = CompiledModel.fromBuffer(bytes);
        try {
          return runCompiledModel(model, 4.0);
        } finally {
          model.close();
        }
      });
      expect(y, closeTo(9.0, 1e-3));
    });

    testWidgets('runAsync works', (tester) async {
      final model = CompiledModel.fromBuffer(modelBytes);
      addTearDown(model.close);
      final input = Float32List(model.inputByteSizes[0] ~/ 4)..[0] = 10.0;
      final output = await model.runAsync([input]);
      expect(output[0][0], closeTo(21.0, 1e-3));
    });

    testWidgets('runs on the Play services GPU when it has one', (
      tester,
    ) async {
      if (status.gpuAvailable != true) {
        markTestSkipped(
          'No Play services GPU accelerator on this device: '
          '${status.gpuError ?? 'none provided'}',
        );
        return;
      }
      final model = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.gpu},
      );
      addTearDown(model.close);
      expect(runCompiledModel(model, 3.0), closeTo(7.0, 1e-2));
    });

    testWidgets('the auto policy compiles and runs', (tester) async {
      final model = CompiledModel.fromBufferWithConfig(modelBytes);
      addTearDown(model.close);
      expect(runCompiledModel(model, 2.0), closeTo(5.0, 1e-2));
    });

    testWidgets('Accelerator.npu throws UnsupportedError', (tester) async {
      expect(
        () => CompiledModel.fromBuffer(
          modelBytes,
          accelerators: {Accelerator.npu},
        ),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('Google Play services'),
          ),
        ),
      );
    });
  });

  group('both runtimes', () {
    testWidgets('run side by side in one process', (tester) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      addTearDown(interpreter.close);
      final model = CompiledModel.fromBuffer(modelBytes);
      addTearDown(model.close);

      for (final x in [-2.0, 0.5, 8.0]) {
        expect(runInterpreter(interpreter, x), closeTo(2 * x + 1, 1e-3));
        expect(runCompiledModel(model, x), closeTo(2 * x + 1, 1e-3));
      }
    });
  });

  group('real models', () => defineRealModelTests(() => status));
}
