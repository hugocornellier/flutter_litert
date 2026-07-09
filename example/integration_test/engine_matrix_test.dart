// ignore_for_file: avoid_print

// Cross-platform engine x accelerator benchmark + correctness matrix.
//
// Runs every bundled model through BOTH runtimes: the classic Interpreter
// (cpu / xnnpack / Metal GPU / GL-CL GPU / CoreML) and the LiteRT Next
// CompiledModel (cpu / strict-GPU fp16+fp32 sync / strict-GPU async / GPU|CPU
// fallback / host-memory async), measuring per cell:
//   - compile_ms   : runtime/delegate build + tensor allocation (cold start)
//   - p50/p90/std  : steady-state inference latency over `iterations` runs
//   - parity       : max abs diff of output 0 vs the Interpreter-CPU reference
//                    (zero inputs), which catches a GPU/driver producing wrong
//                    results rather than just being slow.
//
// Every cell is independently guarded so one bad (model, mode) pair surfaces as
// a status (err / dyn / unsupported) instead of aborting the run, and delegates
// are built DIRECTLY so a failed init shows up instead of silently becoming CPU.
//
// Results are emitted two ways: a human-readable table to stdout, and structured
// long-format rows via `binding.reportData`, which test_driver/integration_test
// .dart appends to test/benchmark/RESULTS.csv on the host (one row per model x
// mode).
//
// Run (profile mode, required for representative numbers) via the wrapper:
//   test/benchmark/run_matrix.sh <device>  # e.g. macos / linux / windows / <id>
// or directly:
//   flutter drive --profile \
//     --driver=test_driver/integration_test.dart \
//     --target=integration_test/engine_matrix_test.dart -d macos

import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter_litert/flutter_litert.dart';

const int iterations = int.fromEnvironment('MATRIX_ITERS', defaultValue: 25);
const int warmup = int.fromEnvironment('MATRIX_WARMUP', defaultValue: 8);

// Fixed, committed benchmark set. Curated to span task type (object / face /
// animal / pose / segmentation), input size, and dtype (fp16 / fp32). Add
// assets to example/pubspec.yaml and a line here to extend.
const models = <String>[
  'assets/mobilefacenet.tflite',
  'assets/species_classifier_float16.tflite',
  'assets/superanimal_rtmpose_s_float16.tflite',
  'assets/yolov8n_float32.tflite',
  'assets/efficientdet_lite0.tflite',
  'assets/selfie_multiclass.tflite',
  'assets/pose_landmark_heavy.tflite',
];

// --- Mode descriptors -------------------------------------------------------

typedef InterpBuild = (InterpreterOptions, Delegate?) Function();

class InterpMode {
  const InterpMode(this.label, this.build);
  final String label;
  final InterpBuild build;
}

final interpModes = <InterpMode>[
  InterpMode('cpu', () => (InterpreterOptions()..threads = 4, null)),
  InterpMode('xnn', () {
    final o = InterpreterOptions()..threads = 4;
    final d = XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: 4));
    o.addDelegate(d);
    return (o, d);
  }),
  InterpMode('gpu_metal', () {
    final o = InterpreterOptions();
    final d = GpuDelegate(
      options: GpuDelegateOptions(allowPrecisionLoss: true),
    );
    o.addDelegate(d);
    return (o, d);
  }),
  InterpMode('gpu_glcl', () {
    final o = InterpreterOptions();
    final d = GpuDelegateV2();
    o.addDelegate(d);
    return (o, d);
  }),
  InterpMode('coreml', () {
    final o = InterpreterOptions();
    final d = CoreMlDelegate(options: CoreMlDelegateOptions(enabledDevices: 1));
    o.addDelegate(d);
    return (o, d);
  }),
];

class CmMode {
  const CmMode(
    this.label,
    this.accelerators,
    this.precision,
    this.bufferMode,
    this.async,
  );
  final String label;
  final Set<Accelerator> accelerators;
  final Precision precision;
  final TensorBufferMode bufferMode;
  final bool async;

  String get accelLabel =>
      (accelerators.toList()..sort((a, b) => a.index.compareTo(b.index)))
          .map((a) => a.name)
          .join('+');
}

final cmModes = <CmMode>[
  CmMode(
    'cm_cpu',
    {Accelerator.cpu},
    Precision.fp32,
    TensorBufferMode.managed,
    false,
  ),
  CmMode(
    'cm_gpu16',
    {Accelerator.gpu},
    Precision.fp16,
    TensorBufferMode.managed,
    false,
  ),
  CmMode(
    'cm_gpu32',
    {Accelerator.gpu},
    Precision.fp32,
    TensorBufferMode.managed,
    false,
  ),
  CmMode(
    'cm_gpuA',
    {Accelerator.gpu},
    Precision.fp16,
    TensorBufferMode.managed,
    true,
  ),
  CmMode(
    'cm_g+c',
    {Accelerator.gpu, Accelerator.cpu},
    Precision.fp32,
    TensorBufferMode.managed,
    false,
  ),
  CmMode(
    'cm_hmA',
    {Accelerator.gpu},
    Precision.fp16,
    TensorBufferMode.hostMemory,
    true,
  ),
];

// --- Per-cell result --------------------------------------------------------

class Cell {
  Cell(
    this.status, {
    this.compileMs,
    this.p50,
    this.p90,
    this.std,
    this.parity,
  });
  final String status; // ok | err | dyn | unsupported
  final double? compileMs, p50, p90, std, parity;

  /// Compact human-readable cell for the stdout table.
  String get pretty {
    if (status != 'ok') return status;
    return '${p50!.toStringAsFixed(0).padLeft(3)}±${std!.toStringAsFixed(0)}';
  }
}

// --- Metric helpers ---------------------------------------------------------

double _pct(List<int> t, double p) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * p).round()].toDouble();
}

double _std(List<int> t) {
  final m = t.reduce((a, b) => a + b) / t.length;
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

double _maxDiff(Float32List ref, List<double> got) {
  final n = min(ref.length, got.length);
  double d = 0;
  for (int i = 0; i < n; i++) {
    final v = (ref[i] - got[i]).abs();
    if (v > d) d = v;
  }
  return d;
}

String _classify(Object e) {
  if (e is UnsupportedError) return 'unsupported';
  final m = e.toString();
  if (m.contains('llocate') || m.contains('precondition')) return 'dyn';
  return 'err';
}

// --- Reference output (Interpreter CPU, zero inputs) ------------------------

Float32List? _refOutput(Uint8List bytes) {
  Interpreter? interp;
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
    final out = interp.getOutputTensor(0).data;
    return Float32List.fromList(
      Float32List.view(out.buffer, out.offsetInBytes, out.lengthInBytes ~/ 4),
    );
  } catch (_) {
    return null;
  } finally {
    interp?.close();
  }
}

// --- Cell runners -----------------------------------------------------------

Cell _interpCell(Uint8List bytes, InterpMode mode, Float32List? ref) {
  Delegate? delegate;
  Interpreter? interp;
  try {
    final sw = Stopwatch()..start();
    final (opts, d) = mode.build();
    delegate = d;
    interp = Interpreter.fromBuffer(bytes, options: opts);
    interp.allocateTensors();
    final compileMs = sw.elapsedMicroseconds / 1000;

    for (int i = 0; i < interp.getInputTensors().length; i++) {
      final t = interp.getInputTensor(i);
      t.data = Uint8List(t.data.length);
    }
    for (int i = 0; i < warmup; i++) {
      interp.invoke();
    }
    final ms = <int>[];
    for (int i = 0; i < iterations; i++) {
      final s = Stopwatch()..start();
      interp.invoke();
      s.stop();
      ms.add((s.elapsedMicroseconds / 1000).round());
    }

    double? parity;
    if (ref != null) {
      final out = interp.getOutputTensor(0).data;
      final got = Float32List.view(
        out.buffer,
        out.offsetInBytes,
        out.lengthInBytes ~/ 4,
      );
      parity = _maxDiff(ref, got);
    }
    return Cell(
      'ok',
      compileMs: compileMs,
      p50: _pct(ms, 0.50),
      p90: _pct(ms, 0.90),
      std: _std(ms),
      parity: parity,
    );
  } catch (e) {
    return Cell(_classify(e));
  } finally {
    interp?.close();
    delegate?.delete();
  }
}

Future<Cell> _cmCell(Uint8List bytes, CmMode mode, Float32List? ref) async {
  CompiledModel? cm;
  try {
    final sw = Stopwatch()..start();
    cm = CompiledModel.fromBuffer(
      bytes,
      accelerators: mode.accelerators,
      precision: mode.precision,
      tensorBufferMode: mode.bufferMode,
    );
    final compileMs = sw.elapsedMicroseconds / 1000;
    final model = cm;
    final inputs = [for (final b in model.inputByteSizes) Float32List(b ~/ 4)];

    Future<List<Float32List>> once() async =>
        mode.async ? model.runAsync(inputs) : model.run(inputs);

    for (int i = 0; i < warmup; i++) {
      await once();
    }
    final ms = <int>[];
    List<Float32List>? last;
    for (int i = 0; i < iterations; i++) {
      final s = Stopwatch()..start();
      last = await once();
      s.stop();
      ms.add((s.elapsedMicroseconds / 1000).round());
    }

    final parity = (ref != null && last != null)
        ? _maxDiff(ref, last[0])
        : null;
    return Cell(
      'ok',
      compileMs: compileMs,
      p50: _pct(ms, 0.50),
      p90: _pct(ms, 0.90),
      std: _std(ms),
      parity: parity,
    );
  } catch (e) {
    return Cell(_classify(e));
  } finally {
    cm?.close();
  }
}

// --- Metadata ---------------------------------------------------------------

Future<Map<String, dynamic>> _collectMeta() async {
  final di = DeviceInfoPlugin();
  var device = 'unknown';
  var extra = '';
  try {
    if (Platform.isMacOS) {
      final i = await di.macOsInfo;
      device = i.model;
      extra = '${i.arch} macOS ${i.osRelease}';
    } else if (Platform.isIOS) {
      final i = await di.iosInfo;
      device = i.utsname.machine;
      extra = '${i.systemName} ${i.systemVersion}';
    } else if (Platform.isAndroid) {
      final i = await di.androidInfo;
      device = '${i.manufacturer} ${i.model}';
      extra = 'SDK ${i.version.sdkInt} (${i.hardware})';
    } else if (Platform.isLinux) {
      final i = await di.linuxInfo;
      device = i.prettyName;
      extra = i.machineId ?? '';
    } else if (Platform.isWindows) {
      final i = await di.windowsInfo;
      device = i.productName;
      extra = 'build ${i.buildNumber}';
    }
  } catch (_) {
    // Best effort; leave defaults.
  }
  return {
    'platform': Platform.operatingSystem,
    'os_version': Platform.operatingSystemVersion,
    'device_model': device,
    'device_extra': extra,
    'build_mode': kReleaseMode
        ? 'release'
        : kProfileMode
        ? 'profile'
        : 'debug',
    'litert_commit': const String.fromEnvironment(
      'LITERT_COMMIT',
      defaultValue: 'unknown',
    ),
    'timestamp': DateTime.now().toUtc().toIso8601String(),
    'iterations': iterations,
    'warmup': warmup,
  };
}

// --- Test driver ------------------------------------------------------------

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final header = [
    ...interpModes.map((m) => m.label),
    ...cmModes.map((m) => m.label),
  ];
  final rows = <Map<String, dynamic>>[]; // long-format, one per (model, mode)
  final prettyRows = <String>[]; // for the stdout table

  group('engine x accelerator matrix', () {
    for (final asset in models) {
      final name = asset.split('/').last.replaceAll('.tflite', '');
      test(name, timeout: const Timeout(Duration(minutes: 20)), () async {
        final data = await rootBundle.load(asset);
        final bytes = data.buffer.asUint8List();
        final modelBytes = bytes.lengthInBytes;
        final ref = _refOutput(bytes);
        final pretty = <String>[];

        void record(
          String engine,
          String mode,
          String accel,
          String precision,
          String buffer,
          bool async,
          Cell c,
        ) {
          rows.add({
            'model_name': name,
            'model_bytes': modelBytes,
            'engine': engine,
            'mode': mode,
            'accelerators': accel,
            'precision': precision,
            'buffer_mode': buffer,
            'async': async,
            'status': c.status,
            'compile_ms': c.compileMs,
            'p50_ms': c.p50,
            'p90_ms': c.p90,
            'std_ms': c.std,
            'parity_maxdiff': c.parity,
          });
          pretty.add(c.pretty);
        }

        for (final m in interpModes) {
          final c = _interpCell(bytes, m, ref);
          record('interpreter', m.label, m.label, '-', '-', false, c);
        }
        for (final m in cmModes) {
          final c = await _cmCell(bytes, m, ref);
          record(
            'compiledmodel',
            m.label,
            m.accelLabel,
            m.precision.name,
            m.bufferMode.name,
            m.async,
            c,
          );
        }

        final row =
            '${name.padRight(30)} ${pretty.map((c) => c.padLeft(9)).join(' ')}';
        prettyRows.add(row);
        print('\n>>> $row\n');
      });
    }

    tearDownAll(() async {
      final meta = await _collectMeta();
      binding.reportData = <String, dynamic>{
        'matrix': {'meta': meta, 'rows': rows},
      };

      print('\n${'=' * 130}');
      print(
        'ENGINE x ACCELERATOR MATRIX: p50±std ms (profile mode for real '
        'numbers). ${meta['platform']} / ${meta['device_model']} / '
        '${meta['build_mode']} / commit ${meta['litert_commit']}',
      );
      print(
        'Interpreter: cpu/xnn/gpu_metal/gpu_glcl/coreml. CompiledModel: '
        'cpu, strict-gpu fp16/fp32 sync, gpu async, gpu|cpu, host-mem async.',
      );
      print('=' * 130);
      print(
        '${'model'.padRight(30)} ${header.map((m) => m.padLeft(9)).join(' ')}',
      );
      print('-' * 130);
      for (final r in prettyRows) {
        print(r);
      }
      print('=' * 130);
      print(
        'Structured rows emitted via reportData -> test/benchmark/RESULTS.csv '
        '(${rows.length} rows).',
      );
    });
  });
}
