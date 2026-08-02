// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';

import 'package:integration_test/integration_test_driver.dart';

const _columns = <String>[
  'timestamp_utc',
  'flutter_litert_commit',
  'platform',
  'platform_version',
  'abi',
  'device_model',
  'device_extra',
  'logical_processors',
  'build_mode',
  'interpreter_runtime_version',
  'iterations',
  'warmup',
  'absolute_tolerance',
  'relative_tolerance',
  'row_started_utc',
  'repository',
  'model_name',
  'model_file',
  'model_path',
  'model_bytes',
  'model_sha256',
  'engine',
  'mode',
  'delegate',
  'accelerators',
  'effective_accelerators',
  'precision',
  'buffer_mode',
  'timing_scope',
  'status',
  'phase',
  'delegate_active',
  'fully_accelerated',
  'accuracy_kind',
  'accuracy_pass',
  'accuracy_cases_passed',
  'accuracy_cases_total',
  'worst_absolute_error',
  'worst_relative_error',
  'worst_tolerance_ratio',
  'compile_ms',
  'first_inference_ms',
  'first_async_inference_ms',
  'sync_samples',
  'sync_min_ms',
  'sync_max_ms',
  'sync_mean_ms',
  'sync_p50_ms',
  'sync_p90_ms',
  'sync_std_ms',
  'async_samples',
  'async_min_ms',
  'async_max_ms',
  'async_mean_ms',
  'async_p50_ms',
  'async_p90_ms',
  'async_std_ms',
  'error_type',
  'error',
];

String _csv(Object? value) {
  if (value == null) return '';
  final text = value is List || value is Map
      ? jsonEncode(value)
      : value.toString();
  if (text.contains(',') || text.contains('"') || text.contains('\n')) {
    return '"${text.replaceAll('"', '""')}"';
  }
  return text;
}

Map<String, Object?> _flatten(
  Map<String, dynamic> meta,
  Map<String, dynamic> row,
) {
  final sync =
      (row['sync_timing'] as Map?)?.cast<String, dynamic>() ?? const {};
  final async =
      (row['async_timing'] as Map?)?.cast<String, dynamic>() ?? const {};
  return {
    ...meta,
    ...row,
    'sync_samples': sync['samples'],
    'sync_min_ms': sync['min_ms'],
    'sync_max_ms': sync['max_ms'],
    'sync_mean_ms': sync['mean_ms'],
    'sync_p50_ms': sync['p50_ms'],
    'sync_p90_ms': sync['p90_ms'],
    'sync_std_ms': sync['std_ms'],
    'async_samples': async['samples'],
    'async_min_ms': async['min_ms'],
    'async_max_ms': async['max_ms'],
    'async_mean_ms': async['mean_ms'],
    'async_p50_ms': async['p50_ms'],
    'async_p90_ms': async['p90_ms'],
    'async_std_ms': async['std_ms'],
  };
}

Future<void> _writeFile(File file, String contents) async {
  await file.parent.create(recursive: true);
  await file.writeAsString(contents, flush: true);
}

String _modelKey(Map<String, dynamic> value) =>
    '${value['repository']}/${value['model_name']}';

String _rowKey(Map<String, dynamic> value) =>
    '${_modelKey(value)}|${value['engine']}|${value['mode']}';

List<Map<String, dynamic>> _mergeByKey(
  List<Map<String, dynamic>> first,
  List<Map<String, dynamic>> second,
  String Function(Map<String, dynamic>) keyOf,
) {
  final values = <String, Map<String, dynamic>>{};
  for (final value in [...first, ...second]) {
    values[keyOf(value)] = value;
  }
  final result = values.values.toList();
  result.sort((a, b) => keyOf(a).compareTo(keyOf(b)));
  return result;
}

Map<String, dynamic> _withSummary(Map<String, dynamic> matrix) {
  final rows = (matrix['rows'] as List)
      .map((row) => (row as Map).cast<String, dynamic>())
      .toList();
  final references = (matrix['references'] as List)
      .map((row) => (row as Map).cast<String, dynamic>())
      .toList();
  final meta = (matrix['meta'] as Map).cast<String, dynamic>();
  final statuses = <String, int>{};
  for (final row in rows) {
    final status = row['status']?.toString() ?? 'missing_status';
    statuses[status] = (statuses[status] ?? 0) + 1;
  }
  final accuracyFailures = rows
      .where((row) => row['status'] == 'ok' && row['accuracy_pass'] != true)
      .length;
  final referenceFailures = references
      .where((reference) => reference['status'] != 'ok')
      .length;
  matrix['summary'] = <String, Object?>{
    'expected_rows': meta['expected_rows'],
    'actual_rows': rows.length,
    'rectangular': rows.length == meta['expected_rows'],
    'status_counts': statuses,
    'reference_failures': referenceFailures,
    'accuracy_failures': accuracyFailures,
    'successful_accuracy_checks': rows
        .where((row) => row['status'] == 'ok' && row['accuracy_pass'] == true)
        .length,
  };
  return matrix;
}

Map<String, dynamic> _mergeMatrices(
  Map<String, dynamic> first,
  Map<String, dynamic> second,
) {
  List<Map<String, dynamic>> list(String key, Map<String, dynamic> source) =>
      (source[key] as List)
          .map((value) => (value as Map).cast<String, dynamic>())
          .toList();

  final firstMeta = (first['meta'] as Map).cast<String, dynamic>();
  final secondMeta = (second['meta'] as Map).cast<String, dynamic>();
  final inventory = _mergeByKey(
    list('inventory', first),
    list('inventory', second),
    _modelKey,
  );
  final references = _mergeByKey(
    list('references', first),
    list('references', second),
    _modelKey,
  );
  final rows = _mergeByKey(list('rows', first), list('rows', second), _rowKey);
  final modes = rows.map((row) => '${row['engine']}|${row['mode']}').toSet();
  final interpreterModes = modes
      .where((mode) => mode.startsWith('interpreter|'))
      .length;
  final compiledModes = modes
      .where((mode) => mode.startsWith('compiled_model|'))
      .length;
  final firstShards =
      (firstMeta['shards'] as List?)?.toList() ??
      [
        {
          'timestamp_utc': firstMeta['timestamp_utc'],
          'model_filter': firstMeta['model_filter'],
          'mode_filter': firstMeta['mode_filter'],
          'expected_rows': firstMeta['expected_rows'],
        },
      ];
  final secondShards =
      (secondMeta['shards'] as List?)?.toList() ??
      [
        {
          'timestamp_utc': secondMeta['timestamp_utc'],
          'model_filter': secondMeta['model_filter'],
          'mode_filter': secondMeta['mode_filter'],
          'expected_rows': secondMeta['expected_rows'],
        },
      ];
  final expectedRows =
      (firstMeta['expected_rows'] as num).toInt() +
      (secondMeta['expected_rows'] as num).toInt();
  final merged = <String, dynamic>{
    'meta': <String, dynamic>{
      ...firstMeta,
      'completed_timestamp_utc': secondMeta['timestamp_utc'],
      'model_count': inventory.length,
      'interpreter_mode_count': interpreterModes,
      'compiled_model_mode_count': compiledModes,
      'expected_rows': expectedRows,
      'model_filter': '',
      'mode_filter': 'process_isolated_shards',
      'shards': [...firstShards, ...secondShards],
    },
    'inventory': inventory,
    'references': references,
    'rows': rows,
  };
  return _withSummary(merged);
}

Future<void> main() => integrationDriver(
  responseDataCallback: (response) async {
    final raw = response?['apple_model_matrix'];
    if (raw is! Map) {
      throw StateError(
        'apple_model_matrix_test did not report a result dataset.',
      );
    }
    var matrix = _withSummary(raw.cast<String, dynamic>());

    // The runner always passes explicit paths. This fallback only covers a
    // bare `flutter drive`, where defaulting an iOS run onto the recorded
    // macOS dataset would silently overwrite it, so derive the destination
    // from the platform the device actually reported.
    final reportedPlatform = ((matrix['meta'] as Map?)?['platform'])
        ?.toString();
    final prefix = reportedPlatform == 'ios' ? 'IOS' : 'MACOS';

    final jsonPath =
        Platform.environment['APPLE_MATRIX_JSON'] ??
        '../test/benchmark/${prefix}_MODEL_MATRIX_RESULTS.json';
    final csvPath =
        Platform.environment['APPLE_MATRIX_CSV'] ??
        '../test/benchmark/${prefix}_MODEL_MATRIX_RESULTS.csv';
    final jsonFile = File(jsonPath);

    if (Platform.environment['APPLE_MATRIX_MERGE'] == 'true' &&
        await jsonFile.exists()) {
      final existing = (jsonDecode(await jsonFile.readAsString()) as Map)
          .cast<String, dynamic>();
      matrix = _mergeMatrices(existing, matrix);
    }

    final meta = (matrix['meta'] as Map).cast<String, dynamic>();
    final rows = (matrix['rows'] as List)
        .map((row) => (row as Map).cast<String, dynamic>())
        .toList();

    final jsonContents =
        '${const JsonEncoder.withIndent('  ').convert(matrix)}\n';
    await _writeFile(jsonFile, jsonContents);

    final csv = StringBuffer()..writeln(_columns.join(','));
    for (final row in rows) {
      final flat = _flatten(meta, row);
      csv.writeln(_columns.map((column) => _csv(flat[column])).join(','));
    }
    await _writeFile(File(csvPath), csv.toString());

    final summary = (matrix['summary'] as Map).cast<String, dynamic>();
    print(
      'apple_model_matrix: wrote ${rows.length} rows to $jsonPath and '
      '$csvPath; status=${summary['status_counts']}, '
      'accuracy_failures=${summary['accuracy_failures']}',
    );
  },
);
