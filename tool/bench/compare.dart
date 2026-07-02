// ignore_for_file: avoid_print
//
// A/B comparison for bench.dart JSONL output.
//
//   dart run tool/bench/compare.dart a.jsonl b.jsonl [--floor-pct=0.5]
//
// Pools all samples per bench per side, then reports the median delta with
// a seeded bootstrap 95% confidence interval on the difference of medians.
// A bench is flagged only when the CI excludes zero AND the median delta
// exceeds the noise floor percentage (set it from an A/A run).

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

const _bootstrapRounds = 10000;

List<double> _pool(String path, Map<String, List<double>> into) {
  final all = <double>[];
  for (final line in File(path).readAsLinesSync()) {
    if (line.trim().isEmpty) continue;
    final row = jsonDecode(line) as Map<String, dynamic>;
    final samples = (row['samples'] as List).cast<num>().map(
      (n) => n.toDouble(),
    );
    into.putIfAbsent(row['bench'] as String, () => []).addAll(samples);
    all.addAll(samples);
  }
  return all;
}

double _median(List<double> sorted) {
  final n = sorted.length;
  if (n.isOdd) return sorted[n ~/ 2];
  return (sorted[n ~/ 2 - 1] + sorted[n ~/ 2]) / 2;
}

double _medianOf(List<double> values) {
  final s = [...values]..sort();
  return _median(s);
}

/// Bootstrap 95% CI of median(b) - median(a).
(double, double) _bootstrapCi(List<double> a, List<double> b, math.Random rng) {
  final diffs = List<double>.filled(_bootstrapRounds, 0);
  final ra = List<double>.filled(a.length, 0);
  final rb = List<double>.filled(b.length, 0);
  for (var i = 0; i < _bootstrapRounds; i++) {
    for (var j = 0; j < a.length; j++) {
      ra[j] = a[rng.nextInt(a.length)];
    }
    for (var j = 0; j < b.length; j++) {
      rb[j] = b[rng.nextInt(b.length)];
    }
    diffs[i] = _medianOf(rb) - _medianOf(ra);
  }
  diffs.sort();
  return (
    diffs[(_bootstrapRounds * 0.025).floor()],
    diffs[(_bootstrapRounds * 0.975).ceil() - 1],
  );
}

void main(List<String> args) {
  final paths = args.where((a) => !a.startsWith('--')).toList();
  if (paths.length != 2) {
    stderr.writeln('Usage: compare.dart a.jsonl b.jsonl [--floor-pct=0.5]');
    exit(64);
  }
  var floorPct = 0.5;
  for (final arg in args) {
    if (arg.startsWith('--floor-pct=')) {
      floorPct = double.parse(arg.substring('--floor-pct='.length));
    }
  }

  final aBenches = <String, List<double>>{};
  final bBenches = <String, List<double>>{};
  _pool(paths[0], aBenches);
  _pool(paths[1], bBenches);

  final rng = math.Random(42);
  var anySignificant = false;

  print(
    '${'bench'.padRight(24)} ${'A median'.padLeft(12)} ${'B median'.padLeft(12)} '
    '${'delta ns'.padLeft(10)} ${'delta %'.padLeft(8)}   95% CI [ns]          verdict',
  );
  for (final bench in aBenches.keys) {
    final a = aBenches[bench];
    final b = bBenches[bench];
    if (a == null || b == null || a.isEmpty || b.isEmpty) continue;
    final ma = _medianOf(a);
    final mb = _medianOf(b);
    final delta = mb - ma;
    final deltaPct = 100 * delta / ma;
    final (lo, hi) = _bootstrapCi(a, b, rng);
    final excludesZero = lo > 0 || hi < 0;
    final clearsFloor = deltaPct.abs() >= floorPct;
    final String verdict;
    if (excludesZero && clearsFloor) {
      verdict = delta < 0 ? 'B FASTER' : 'B SLOWER';
      anySignificant = true;
    } else {
      verdict = 'no significant difference';
    }
    print(
      '${bench.padRight(24)} ${ma.toStringAsFixed(1).padLeft(12)} ${mb.toStringAsFixed(1).padLeft(12)} '
                  '${delta.toStringAsFixed(1).padLeft(10)} ${deltaPct.toStringAsFixed(2).padLeft(7)}%   '
                  '[${lo.toStringAsFixed(1)}, ${hi.toStringAsFixed(1)}]'
              .padRight(24) +
          verdict,
    );
  }
  if (!anySignificant) {
    print(
      'overall: no bench cleared the ${floorPct.toStringAsFixed(2)}% floor with CI excluding zero',
    );
  }
}
