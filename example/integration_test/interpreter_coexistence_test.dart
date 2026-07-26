import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

/// On-device coverage for the classic [Interpreter], and for both native
/// runtimes being live in one process at the same time.
///
/// The package ships two independent runtimes: classic TFLite
/// (libtensorflowlite_c on desktop, libtensorflowlite_jni on Android, the
/// TensorFlowLiteC framework on iOS) and LiteRT Next (libLiteRt). They export
/// overlapping TFLite symbols, so whichever one is dlopen'd first can end up
/// satisfying the other's relocations. Host unit tests cannot catch that: they
/// run on macOS and Linux only, and each suite touches a single runtime per
/// process. Everything here runs inside a real app process, brings the two
/// runtimes up in both orders, and asserts each keeps producing its own
/// correct results afterwards.
///
/// simple_model is y = 2*x + 1, so the cross-runtime comparisons are exact
/// rather than a tolerance on a real network's output. The real-network case
/// below deliberately asserts structure and finiteness instead of numeric
/// parity: the two runtimes use different kernel libraries, so a tight
/// tolerance there would be measuring kernel accuracy, not coexistence.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List modelBytes;
  late Uint8List segModelBytes;

  setUpAll(() async {
    modelBytes = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
    segModelBytes = (await rootBundle.load(
      'assets/selfie_multiclass.tflite',
    )).buffer.asUint8List();
  });

  /// Runs simple_model on [interpreter] through the tensor views, the closest
  /// analogue to the flat-buffer API CompiledModel exposes.
  double runInterpreter(Interpreter interpreter, double x) {
    interpreter.getInputTensor(0).asFloat32View()[0] = x;
    interpreter.invoke();
    return interpreter.getOutputTensor(0).asFloat32View()[0];
  }

  double runCompiledModel(CompiledModel cm, double x) {
    final input = Float32List(cm.inputByteSizes[0] ~/ 4);
    input[0] = x;
    return cm.run([input])[0][0];
  }

  group('classic Interpreter on device', () {
    testWidgets('runs and returns correct results', (tester) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      addTearDown(interpreter.close);

      expect(interpreter.getInputTensors(), isNotEmpty);
      expect(interpreter.getOutputTensors(), isNotEmpty);
      expect(runInterpreter(interpreter, 3.0), closeTo(7.0, 1e-3));
      expect(runInterpreter(interpreter, 0.0), closeTo(1.0, 1e-3));
      expect(runInterpreter(interpreter, -1.0), closeTo(-1.0, 1e-3));
    });

    testWidgets('runs a real network from the app bundle', (tester) async {
      // simple_model is a single op. A real graph exercises far more of the
      // classic runtime's kernel set, and a large output tensor exercises the
      // read-back path rather than a single float.
      final interpreter = Interpreter.fromBuffer(segModelBytes);
      addTearDown(interpreter.close);

      final input = interpreter.getInputTensor(0).asFloat32View();
      input.fillRange(0, input.length, 0);
      interpreter.invoke();

      final output = interpreter.getOutputTensor(0).asFloat32View();
      expect(output.length, greaterThan(1 << 18));
      expect(output.every((v) => v.isFinite), isTrue);
    });

    testWidgets('auto PerformanceConfig yields a working interpreter', (
      tester,
    ) async {
      // Covers the platform default delegate path (Metal on Apple, XNNPack
      // elsewhere) including the CPU retry when a model cannot be delegated.
      final (options, delegate) = InterpreterFactory.create(
        const PerformanceConfig(),
      );
      final interpreter = Interpreter.fromBuffer(modelBytes, options: options);
      addTearDown(() {
        interpreter.close();
        options.delete();
        delegate?.delete();
      });

      expect(runInterpreter(interpreter, 3.0), closeTo(7.0, 1e-3));
      expect(interpreter.hasActiveDelegate, isA<bool>());
    });
  });

  group('Interpreter and CompiledModel in one process', () {
    testWidgets('CompiledModel loaded second; both keep working', (
      tester,
    ) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      addTearDown(interpreter.close);
      expect(runInterpreter(interpreter, 3.0), closeTo(7.0, 1e-3));

      // Bringing the second runtime up must not disturb the first: re-running
      // the interpreter afterwards is the assertion that matters here.
      final cm = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(cm.close);

      expect(runCompiledModel(cm, 3.0), closeTo(7.0, 1e-3));
      expect(runInterpreter(interpreter, 5.0), closeTo(11.0, 1e-3));
    });

    testWidgets('Interpreter loaded second; both keep working', (tester) async {
      final cm = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(cm.close);
      expect(runCompiledModel(cm, 3.0), closeTo(7.0, 1e-3));

      final interpreter = Interpreter.fromBuffer(modelBytes);
      addTearDown(interpreter.close);

      expect(runInterpreter(interpreter, 3.0), closeTo(7.0, 1e-3));
      expect(runCompiledModel(cm, 5.0), closeTo(11.0, 1e-3));
    });

    testWidgets('interleaved runs stay independent', (tester) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      final cm = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(interpreter.close);
      addTearDown(cm.close);

      // Alternating keeps both runtimes' tensor arenas hot simultaneously; a
      // shared-state or symbol-collision problem shows up as one side drifting
      // rather than as a load-time failure.
      for (var i = 0; i < 10; i++) {
        final x = i.toDouble();
        final expected = 2 * x + 1;
        expect(runInterpreter(interpreter, x), closeTo(expected, 1e-3));
        expect(runCompiledModel(cm, x), closeTo(expected, 1e-3));
      }
    });

    testWidgets('closing one runtime leaves the other usable', (tester) async {
      final interpreter = Interpreter.fromBuffer(modelBytes);
      final cm = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(interpreter.close);

      expect(runCompiledModel(cm, 3.0), closeTo(7.0, 1e-3));
      cm.close();
      expect(runInterpreter(interpreter, 3.0), closeTo(7.0, 1e-3));
    });

    testWidgets('both runtimes run a real network side by side', (
      tester,
    ) async {
      final interpreter = Interpreter.fromBuffer(segModelBytes);
      final cm = CompiledModel.fromBuffer(
        segModelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(interpreter.close);
      addTearDown(cm.close);

      final input = interpreter.getInputTensor(0).asFloat32View();
      input.fillRange(0, input.length, 0);
      interpreter.invoke();
      final interpreterOut = interpreter.getOutputTensor(0).asFloat32View();

      final cmOut = cm.run([Float32List(cm.inputByteSizes[0] ~/ 4)])[0];

      // Structural agreement only: both runtimes must see the same graph and
      // produce usable numbers. See the file comment for why this is not a
      // tolerance check.
      expect(cmOut, hasLength(interpreterOut.length));
      expect(interpreterOut.every((v) => v.isFinite), isTrue);
      expect(cmOut.every((v) => v.isFinite), isTrue);
    });
  });
}
