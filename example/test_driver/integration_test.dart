// ignore_for_file: avoid_print

import 'dart:io';

import 'package:integration_test/integration_test_driver.dart';

// Long-format CSV columns. Metadata (shared per run) is flattened into every
// row so each line stands alone.
const _columns = <String>[
  'timestamp',
  'litert_commit',
  'platform',
  'os_version',
  'device_model',
  'device_extra',
  'build_mode',
  'iterations',
  'warmup',
  'model_name',
  'model_bytes',
  'engine',
  'mode',
  'accelerators',
  'precision',
  'buffer_mode',
  'async',
  'status',
  'compile_ms',
  'p50_ms',
  'p90_ms',
  'std_ms',
  'parity_maxdiff',
];

String _csv(Object? v) {
  if (v == null) return '';
  final s = v.toString();
  if (s.contains(',') || s.contains('"') || s.contains('\n')) {
    return '"${s.replaceAll('"', '""')}"';
  }
  return s;
}

// Appends the matrix rows from the on-device test to test/benchmark/RESULTS.csv
// on the host. The output path comes from MATRIX_CSV (set by
// test/benchmark/run_matrix.sh); defaults to RESULTS.csv in the current working
// directory.
Future<void> main() => integrationDriver(
  responseDataCallback: (data) async {
    final matrix = data?['matrix'] as Map<String, dynamic>?;
    if (matrix == null) {
      print('engine_matrix: no matrix data reported; nothing written.');
      return;
    }
    final meta = (matrix['meta'] as Map).cast<String, dynamic>();
    final rows = (matrix['rows'] as List).cast<Map>();

    final path = Platform.environment['MATRIX_CSV'] ?? 'RESULTS.csv';
    final file = File(path);
    final exists = await file.exists();
    final sink = file.openWrite(mode: FileMode.append);
    if (!exists) sink.writeln(_columns.join(','));

    for (final row in rows) {
      final r = row.cast<String, dynamic>();
      sink.writeln(
        _columns
            .map((c) {
              final v = meta.containsKey(c) ? meta[c] : r[c];
              return _csv(v);
            })
            .join(','),
      );
    }
    await sink.flush();
    await sink.close();
    print('engine_matrix: appended ${rows.length} rows to $path');
  },
);
