@TestOn('mac-os || linux || windows')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Dart-side overhead benchmark for SignatureRunner.run().
///
/// The training model's signatures are 1x1 matmuls, so native inference time
/// is negligible and the timings are dominated by per-call Dart/FFI overhead
/// (tensor lookups by name, shape checks, staging, output copies).
///
/// Run with:
///   flutter test test/benchmark/signature_runner_benchmark_test.dart
const _model = 'test/assets/training_model.tflite';
const _warmup = 50;
const _iters = 500;

class _Stats {
  _Stats(this.name, List<int> micros)
    : median = _percentile(micros, 50),
      p90 = _percentile(micros, 90);

  final String name;
  final int median;
  final int p90;

  static int _percentile(List<int> v, int p) {
    final s = [...v]..sort();
    return s[(s.length * p ~/ 100).clamp(0, s.length - 1)];
  }

  @override
  String toString() =>
      '${name.padRight(28)} median ${'$median'.padLeft(7)} µs   '
      'p90 ${'$p90'.padLeft(7)} µs';
}

_Stats _bench(String name, void Function() body) {
  for (var i = 0; i < _warmup; i++) {
    body();
  }
  final times = <int>[];
  final sw = Stopwatch();
  for (var i = 0; i < _iters; i++) {
    sw
      ..reset()
      ..start();
    body();
    sw.stop();
    times.add(sw.elapsedMicroseconds);
  }
  return _Stats(name, times);
}

void main() {
  test('signature runner overhead benchmark', () {
    final interpreter = Interpreter.fromFile(File(_model));
    final results = <_Stats>[];

    final inferRunner = interpreter.getSignatureRunner('infer');
    final x = Float32List.fromList([0.5]);
    final output = Float32List(1);
    results.add(
      _bench('infer run()', () {
        inferRunner.run({'x': x}, {'output': output});
      }),
    );
    expect(output[0].isFinite, isTrue);
    inferRunner.close();

    final trainRunner = interpreter.getSignatureRunner('train');
    final y = Float32List.fromList([1.0]);
    final loss = Float32List(1);
    results.add(
      _bench('train run()', () {
        trainRunner.run({'x': x, 'y': y}, {'loss': loss});
      }),
    );
    expect(loss[0].isFinite, isTrue);
    trainRunner.close();

    interpreter.close();

    // ignore: avoid_print
    results.forEach(print);
  });
}
