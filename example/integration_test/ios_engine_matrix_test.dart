// ignore_for_file: avoid_print, deprecated_member_use

// On-device engine x accelerator matrix: every bundled model through BOTH
// runtimes: classic Interpreter (cpu / xnnpack / Metal GPU delegate / CoreML)
// and LiteRT Next CompiledModel (cpu / strict-GPU sync / strict-GPU async /
// GPU|CPU fallback / GPU host-memory async), plus a CompiledModel-vs-
// Interpreter output parity column (max abs diff on output 0, zero inputs).
//
// Delegates and accelerators are built DIRECTLY so a failure to initialize
// shows as ERR in that cell instead of silently becoming CPU. Cells are
// independently guarded so one bad (model, mode) pair doesn't abort the rest.
//
//   flutter test integration_test/ios_engine_matrix_test.dart -d <ios-device-id>

import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

const int iterations = 25;
const int warmup = 8;

const models = <String>[
  'assets/mobilefacenet.tflite',
  'assets/species_classifier_float16.tflite',
  'assets/superanimal_rtmpose_s_float16.tflite',
  'assets/yolov8n_float32.tflite',
  'assets/efficientdet_lite0.tflite',
  'assets/selfie_multiclass.tflite',
  'assets/pose_landmark_heavy.tflite',
];

final interpreterModes = <String, (InterpreterOptions, Delegate?) Function()>{
  'cpu': () => (InterpreterOptions()..threads = 4, null),
  'xnn': () {
    final o = InterpreterOptions()..threads = 4;
    final d = XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: 4));
    o.addDelegate(d);
    return (o, d);
  },
  'gpu': () {
    final o = InterpreterOptions();
    final d = GpuDelegate(
      options: GpuDelegateOptions(allowPrecisionLoss: true),
    );
    o.addDelegate(d);
    return (o, d);
  },
  'coreml': () {
    final o = InterpreterOptions();
    final d = CoreMlDelegate(options: CoreMlDelegateOptions(enabledDevices: 1));
    o.addDelegate(d);
    return (o, d);
  },
};

// (label, accelerators, async dispatch, tensor buffer mode)
final compiledModes = <(String, Set<Accelerator>, bool, TensorBufferMode)>[
  ('cm_cpu', {Accelerator.cpu}, false, TensorBufferMode.managed),
  ('cm_gpu', {Accelerator.gpu}, false, TensorBufferMode.managed),
  ('cm_gpuA', {Accelerator.gpu}, true, TensorBufferMode.managed),
  (
    'cm_g+c',
    {Accelerator.gpu, Accelerator.cpu},
    false,
    TensorBufferMode.managed,
  ),
  ('cm_hmA', {Accelerator.gpu}, true, TensorBufferMode.hostMemory),
];

double _p50(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.50).floor()].toDouble();
}

double _std(List<int> t) {
  final m = t.reduce((a, b) => a + b) / t.length;
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

String _fmt(List<int> ms) =>
    '${_p50(ms).toStringAsFixed(0).padLeft(4)}±${_std(ms).toStringAsFixed(0)}';

String _interpCell(
  Uint8List bytes,
  (InterpreterOptions, Delegate?) Function() build,
) {
  Delegate? delegate;
  Interpreter? interp;
  try {
    final r = build();
    delegate = r.$2;
    interp = Interpreter.fromBuffer(bytes, options: r.$1);
    interp.allocateTensors();
    for (int i = 0; i < warmup; i++) {
      interp.invoke();
    }
    final ms = <int>[];
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      interp.invoke();
      sw.stop();
      ms.add((sw.elapsedMicroseconds / 1000).round());
    }
    return _fmt(ms);
  } catch (e) {
    final msg = e.toString();
    if (msg.contains('llocate') || msg.contains('precondition')) return ' DYN ';
    return ' ERR ';
  } finally {
    interp?.close();
    delegate?.delete();
  }
}

Future<String> _compiledCell(
  Uint8List bytes,
  Set<Accelerator> accelerators,
  bool useAsync,
  TensorBufferMode bufferMode,
) async {
  CompiledModel? cm;
  try {
    cm = CompiledModel.fromBuffer(
      bytes,
      accelerators: accelerators,
      tensorBufferMode: bufferMode,
    );
    final model = cm;
    final inputs = [for (final b in model.inputByteSizes) Float32List(b ~/ 4)];
    Future<void> once() async {
      if (useAsync) {
        await model.runAsync(inputs);
      } else {
        model.run(inputs);
      }
    }

    for (int i = 0; i < warmup; i++) {
      await once();
    }
    final ms = <int>[];
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      await once();
      sw.stop();
      ms.add((sw.elapsedMicroseconds / 1000).round());
    }
    return _fmt(ms);
  } catch (e) {
    return ' ERR ';
  } finally {
    cm?.close();
  }
}

// Max abs diff between Interpreter-CPU and CompiledModel-CPU on output 0 with
// all-zero inputs. Informational: multi-output models may legitimately order
// outputs differently between the two runtimes.
Future<String> _parity(Uint8List bytes) async {
  Interpreter? interp;
  CompiledModel? cm;
  try {
    interp = Interpreter.fromBuffer(
      bytes,
      options: InterpreterOptions()..threads = 4,
    );
    interp.allocateTensors();
    for (int i = 0; i < interp.getInputTensors().length; i++) {
      final t = interp.getInputTensor(i);
      t.data = Uint8List(t.data.length);
    }
    interp.invoke();
    final outBytes = interp.getOutputTensor(0).data;
    final ref = Float32List.view(
      outBytes.buffer,
      outBytes.offsetInBytes,
      outBytes.lengthInBytes ~/ 4,
    );

    cm = CompiledModel.fromBuffer(bytes);
    final got = cm.run([
      for (final b in cm.inputByteSizes) Float32List(b ~/ 4),
    ])[0];

    final n = min(ref.length, got.length);
    double maxDiff = 0;
    for (int i = 0; i < n; i++) {
      final d = (ref[i] - got[i]).abs();
      if (d > maxDiff) maxDiff = d;
    }
    return maxDiff.toStringAsExponential(0);
  } catch (e) {
    return 'ERR';
  } finally {
    interp?.close();
    cm?.close();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final interpNames = interpreterModes.keys.toList();
  final cmNames = compiledModes.map((m) => m.$1).toList();
  final header = [...interpNames, ...cmNames, 'parity'];
  final rows = <String>[];

  group('iOS engine x accelerator matrix', () {
    for (final asset in models) {
      final name = asset.split('/').last.replaceAll('.tflite', '');
      test(name, timeout: const Timeout(Duration(minutes: 15)), () async {
        final data = await rootBundle.load(asset);
        final bytes = data.buffer.asUint8List();
        final cells = <String>[];
        for (final mode in interpNames) {
          cells.add(_interpCell(bytes, interpreterModes[mode]!));
        }
        for (final (_, accel, useAsync, bufferMode) in compiledModes) {
          cells.add(await _compiledCell(bytes, accel, useAsync, bufferMode));
        }
        cells.add(await _parity(bytes));
        final row =
            '${name.padRight(30)} ${cells.map((c) => c.padLeft(8)).join(' ')}';
        rows.add(row);
        print('\n>>> $row\n');
      });
    }

    tearDownAll(() {
      print('\n${'=' * 120}');
      print(
        'iOS ENGINE x ACCELERATOR MATRIX: p50±std ms. '
        'Interpreter: cpu/xnn/gpu(Metal)/coreml(ANE). '
        'CompiledModel: cpu, strict-gpu sync/async, gpu|cpu, host-memory async. '
        'parity = max abs diff CM-cpu vs interp-cpu (output 0, zero inputs).',
      );
      print('=' * 120);
      print(
        '${'model'.padRight(30)} ${header.map((m) => m.padLeft(8)).join(' ')}',
      );
      print('-' * 120);
      for (final r in rows) {
        print(r);
      }
      print('=' * 120);
    });
  });
}
