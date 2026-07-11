// ignore_for_file: avoid_print, deprecated_member_use

// On-device iOS delegate sweep. Loads bundled models via rootBundle (the
// device has no access to the Mac filesystem), builds each delegate DIRECTLY
// so a failure to load/apply throws here instead of silently becoming CPU.
//
// This is the only configuration that can actually exercise the Apple Neural
// Engine via the CoreML delegate. coreml uses enabledDevices=AllDevices(1),
// matching what InterpreterFactory ships.
//
//   flutter test integration_test/ios_delegate_sweep_test.dart -d <ios-device-id>

import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
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

final modeBuilders = <String, (InterpreterOptions, Delegate?) Function()>{
  'cpu': () => (InterpreterOptions()..threads = 4, null),
  'xnnpack': () {
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

double _p50(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.50).floor()].toDouble();
}

double _std(List<int> t) {
  final m = t.reduce((a, b) => a + b) / t.length;
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

String _benchCell(
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
    final us = <int>[];
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      interp.invoke();
      sw.stop();
      us.add(sw.elapsedMicroseconds);
    }
    final ms = us.map((u) => (u / 1000).round()).toList();
    return '${_p50(ms).toStringAsFixed(0).padLeft(4)}±${_std(ms).toStringAsFixed(0)}';
  } catch (e) {
    final msg = e.toString();
    if (msg.contains('llocate') || msg.contains('precondition')) return ' DYN ';
    return ' ERR ';
  } finally {
    interp?.close();
    delegate?.delete();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final modeNames = modeBuilders.keys.toList();
  final rows = <String>[];

  group('iOS on-device delegate sweep', () {
    for (final asset in models) {
      final name = asset.split('/').last.replaceAll('.tflite', '');
      test(name, timeout: const Timeout(Duration(minutes: 15)), () async {
        final data = await rootBundle.load(asset);
        final bytes = data.buffer.asUint8List();
        final cells = <String>[];
        for (final mode in modeNames) {
          cells.add(_benchCell(bytes, modeBuilders[mode]!));
        }
        final row =
            '${name.padRight(30)} ${cells.map((c) => c.padLeft(8)).join(' ')}';
        rows.add(row);
        print('\n>>> $row\n');
      });
    }

    tearDownAll(() {
      print('\n${'=' * 80}');
      print('iOS ON-DEVICE DELEGATE SWEEP: p50±std ms (real ANE available)');
      print('=' * 80);
      print(
        '${'model'.padRight(30)} ${modeNames.map((m) => m.padLeft(8)).join(' ')}',
      );
      print('-' * 80);
      for (final r in rows) {
        print(r);
      }
      print('=' * 80);
    });
  });
}
