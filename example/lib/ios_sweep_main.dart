// ignore_for_file: avoid_print, deprecated_member_use

// Standalone on-device delegate sweep, runnable via `flutter run` (avoids the
// integration_test VM-service handshake that fails on physical iOS devices).
//
//   flutter run -d <ios-device-id> -t lib/ios_sweep_main.dart
//
// Watch the console for ">>>" rows and the final "SWEEP DONE" summary.

import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
  String label,
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
  } catch (e, st) {
    final oneLine = e.toString().replaceAll('\n', ' ');
    print('EXC[$label]: $oneLine');
    print('EXC-ST[$label]: ${st.toString().split('\n').take(3).join(' | ')}');
    final msg = e.toString();
    if (msg.contains('llocate') || msg.contains('precondition')) return ' DYN ';
    return ' ERR ';
  } finally {
    interp?.close();
    delegate?.delete();
  }
}

Future<List<String>> _runSweep() async {
  final modeNames = modeBuilders.keys.toList();
  final rows = <String>[];
  print('\n=== iOS ON-DEVICE SWEEP START ===');
  print(
    '${'model'.padRight(30)} ${modeNames.map((m) => m.padLeft(8)).join(' ')}',
  );
  for (final asset in models) {
    final name = asset.split('/').last.replaceAll('.tflite', '');
    final data = await rootBundle.load(asset);
    final bytes = data.buffer.asUint8List();
    final cells = <String>[];
    for (final mode in modeNames) {
      cells.add(_benchCell('$mode/$name', bytes, modeBuilders[mode]!));
    }
    final row =
        '${name.padRight(30)} ${cells.map((c) => c.padLeft(8)).join(' ')}';
    rows.add(row);
    print('>>> $row');
    stderr.writeln('>>> $row');
  }
  print('\n=== SWEEP DONE (real ANE available) ===');
  print(
    '${'model'.padRight(30)} ${modeNames.map((m) => m.padLeft(8)).join(' ')}',
  );
  for (final r in rows) {
    print(r);
  }
  print('=== END ===');
  stderr.writeln('=== END ===');
  return rows;
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _SweepApp());
}

class _SweepApp extends StatefulWidget {
  const _SweepApp();
  @override
  State<_SweepApp> createState() => _SweepAppState();
}

class _SweepAppState extends State<_SweepApp> {
  List<String> _rows = const [];
  bool _done = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final rows = await _runSweep();
      if (mounted) {
        setState(() {
          _rows = rows;
          _done = true;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text(_done ? 'Sweep done' : 'Running sweep…')),
        body: ListView(
          children: [
            const Padding(
              padding: EdgeInsets.all(8),
              child: Text(
                'model            cpu xnnpack gpu coreml',
                style: TextStyle(fontFamily: 'monospace', fontSize: 11),
              ),
            ),
            for (final r in _rows)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                child: Text(
                  r,
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
                ),
              ),
            if (!_done)
              const Padding(
                padding: EdgeInsets.all(16),
                child: Center(child: CircularProgressIndicator()),
              ),
          ],
        ),
      ),
    );
  }
}
