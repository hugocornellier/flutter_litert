@TestOn('mac-os || linux || windows')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// End-to-end engine overhead benchmark on a real per-frame model
/// (MediaPipe face_detection_short_range: input 1x128x128x3 float32,
/// outputs 1x896x16 regressors + 1x896x1 scores).
///
/// Each variant performs the same logical work: stage one input frame, run
/// inference, materialize both outputs. The difference between a variant
/// and the `views floor` variant is pure Dart-side API overhead.
///
/// Run with:
///   flutter test test/benchmark/engine_overhead_benchmark_test.dart
const _model = 'test/assets/face_detection_short_range.tflite';
const _warmup = 20;
const _iters = 100;

const _inLen = 128 * 128 * 3;
const _boxesLen = 896 * 16;
const _scoresLen = 896 * 1;

class _Stats {
  _Stats(this.name, List<int> micros, {this.note = ''})
    : median = _percentile(micros, 50),
      p90 = _percentile(micros, 90);

  _Stats.failed(this.name, this.note) : median = -1, p90 = -1;

  final String name;
  final int median;
  final int p90;
  final String note;

  static int _percentile(List<int> v, int p) {
    final s = [...v]..sort();
    return s[(s.length * p ~/ 100).clamp(0, s.length - 1)];
  }

  @override
  String toString() => median < 0
      ? '${name.padRight(28)} $note'
      : '${name.padRight(28)} median ${'$median'.padLeft(7)} µs   '
            'p90 ${'$p90'.padLeft(7)} µs   $note';
}

_Stats _bench(String name, void Function() body, {String note = ''}) {
  try {
    body(); // Probe: fail fast before timing.
  } catch (e) {
    return _Stats.failed(name, 'FAILS: ${e.toString().split('\n').first}');
  }
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
  return _Stats(name, times, note: note);
}

Float32List _makeInput() {
  final rng = math.Random(42);
  final input = Float32List(_inLen);
  for (var i = 0; i < input.length; i++) {
    input[i] = rng.nextDouble() * 2 - 1;
  }
  return input;
}

List<List<List<List<double>>>> _nestedInput(Float32List flat) => [
  [
    for (var y = 0; y < 128; y++)
      [
        for (var x = 0; x < 128; x++)
          [for (var c = 0; c < 3; c++) flat[(y * 128 + x) * 3 + c]],
      ],
  ],
];

double _maxAbsDiff(Float32List a, Float32List b) {
  var m = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > m) m = d;
  }
  return m;
}

void _flattenNested3(List nested, Float32List out) {
  var i = 0;
  for (final row in nested[0] as List) {
    for (final v in row as List) {
      out[i++] = (v as num).toDouble();
    }
  }
}

void main() {
  final input = _makeInput();
  final refBoxes = Float32List(_boxesLen);
  final refScores = Float32List(_scoresLen);

  /// Runs one interpreter-path variant against a fresh interpreter so a
  /// failing variant cannot corrupt the others' state.
  _Stats interpreterVariant(
    String name,
    void Function(Interpreter interpreter) body, {
    void Function()? verify,
  }) {
    final interpreter = Interpreter.fromFile(File(_model));
    interpreter.allocateTensors();
    final stats = _bench(name, () => body(interpreter));
    if (stats.median >= 0 && verify != null) {
      verify();
    }
    interpreter.close();
    return stats;
  }

  test('engine overhead benchmark', () {
    final results = <_Stats>[];
    final gotBoxes = Float32List(_boxesLen);

    // --- Reference: cached tensor views + invoke (the floor). ---
    {
      final interpreter = Interpreter.fromFile(File(_model));
      interpreter.allocateTensors();
      final views = TensorFloat32Views.capture(interpreter);
      results.add(
        _bench('views floor', () {
          views.inputs[0].setAll(0, input);
          interpreter.invoke();
          refBoxes.setAll(0, views.outputs[0]);
          refScores.setAll(0, views.outputs[1]);
        }),
      );
      interpreter.close();
    }

    final nestedBoxesOut = [
      List.generate(896, (_) => List<double>.filled(16, 0)),
    ];
    final nestedScoresOut = [
      List.generate(896, (_) => List<double>.filled(1, 0)),
    ];
    final nestedIn = _nestedInput(input);
    final flatBoxesOut = Float32List(_boxesLen);
    final flatScoresOut = Float32List(_scoresLen);

    results.add(
      interpreterVariant(
        'run nested-in nested-out',
        (interpreter) => interpreter.runForMultipleInputs(
          [nestedIn],
          {0: nestedBoxesOut, 1: nestedScoresOut},
        ),
        verify: () {
          _flattenNested3(nestedBoxesOut, gotBoxes);
          expect(_maxAbsDiff(gotBoxes, refBoxes), lessThan(1e-5));
        },
      ),
    );

    results.add(
      interpreterVariant(
        'run bytebuffer-in nested-out',
        (interpreter) => interpreter.runForMultipleInputs(
          [input.buffer],
          {0: nestedBoxesOut, 1: nestedScoresOut},
        ),
        verify: () {
          _flattenNested3(nestedBoxesOut, gotBoxes);
          expect(_maxAbsDiff(gotBoxes, refBoxes), lessThan(1e-5));
        },
      ),
    );

    results.add(
      interpreterVariant(
        'run f32-in nested-out',
        (interpreter) => interpreter.runForMultipleInputs(
          [input],
          {0: nestedBoxesOut, 1: nestedScoresOut},
        ),
        verify: () {
          _flattenNested3(nestedBoxesOut, gotBoxes);
          expect(_maxAbsDiff(gotBoxes, refBoxes), lessThan(1e-5));
        },
      ),
    );

    results.add(
      interpreterVariant(
        'run f32-in f32-out',
        (interpreter) => interpreter.runForMultipleInputs(
          [input],
          {0: flatBoxesOut, 1: flatScoresOut},
        ),
        verify: () {
          expect(_maxAbsDiff(flatBoxesOut, refBoxes), lessThan(1e-5));
          expect(_maxAbsDiff(flatScoresOut, refScores), lessThan(1e-5));
        },
      ),
    );

    // --- CompiledModel CPU, managed buffers. ---
    var cmDiff = double.nan;
    {
      final cm = CompiledModel.fromFile(
        _model,
        accelerators: {Accelerator.cpu},
      );
      late List<Float32List> cmOut;
      results.add(
        _bench('compiledmodel managed', () => cmOut = cm.run([input])),
      );
      cmDiff = _maxAbsDiff(cmOut[0], refBoxes);
      cm.close();
    }

    // --- CompiledModel CPU, host-memory direct I/O. ---
    var hostDiff = double.nan;
    {
      final cm = CompiledModel.fromFile(
        _model,
        accelerators: {Accelerator.cpu},
        tensorBufferMode: TensorBufferMode.hostMemory,
      );
      final hostBoxes = Float32List(_boxesLen);
      results.add(
        _bench('compiledmodel hostmem', () {
          cm.writeInput(0, (dst) => dst.setAll(0, input));
          cm.dispatch();
          cm.readOutput(0, (out) => hostBoxes.setAll(0, out));
          cm.readOutput(1, (out) {});
        }),
      );
      hostDiff = _maxAbsDiff(hostBoxes, refBoxes);
      cm.close();
    }

    // ignore: avoid_print
    print('\n=== engine overhead benchmark ($_iters iters) ===');
    for (final r in results) {
      // ignore: avoid_print
      print(r);
    }
    // ignore: avoid_print
    print(
      'compiledmodel-vs-interpreter max-abs-diff: '
      'managed=${cmDiff.toStringAsExponential(2)} '
      'hostmem=${hostDiff.toStringAsExponential(2)}\n',
    );
    expect(cmDiff, lessThan(1e-2));
    expect(hostDiff, lessThan(1e-2));
  }, timeout: const Timeout(Duration(minutes: 5)));
}
