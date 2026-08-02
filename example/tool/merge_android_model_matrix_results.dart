import 'dart:convert';
import 'dart:io';

import 'android_model_matrix_manifest.dart';

Never _usage([String? message]) {
  if (message != null) stderr.writeln('Error: $message');
  stderr.writeln(
    'Usage: dart run example/tool/merge_android_model_matrix_results.dart '
    '--input-root <downloaded Test Lab artifacts> --output-dir <dir> '
    '[--source-run <id>]',
  );
  exit(64);
}

String _requiredOption(List<String> args, String name) {
  final index = args.indexOf(name);
  if (index < 0 || index + 1 >= args.length) _usage('Missing $name.');
  return args[index + 1];
}

String? _optionalOption(List<String> args, String name) {
  final index = args.indexOf(name);
  if (index < 0) return null;
  if (index + 1 >= args.length) _usage('Missing value after $name.');
  return args[index + 1];
}

class _RecordChunks {
  _RecordChunks(this.total);

  final int total;
  final Map<int, String> parts = {};
  bool conflict = false;

  void add(int part, int declaredTotal, String data) {
    if (declaredTotal != total || part < 1 || part > total) {
      conflict = true;
      return;
    }
    final previous = parts[part];
    if (previous != null && previous != data) conflict = true;
    parts[part] = data;
  }

  bool get complete => !conflict && parts.length == total;

  String join() =>
      [for (var index = 1; index <= total; index++) parts[index]!].join();
}

String _cellKey(String model, String mode) => '$model|$mode';

int? _shardFromPath(String path) {
  // "shard-all" is the single-execution run that covers every model.
  if (RegExp(r'shard[-_/]all\b').hasMatch(path)) return androidMatrixAllShards;
  final match = RegExp(r'shard[-_/](\d+)').firstMatch(path);
  return match == null ? null : int.tryParse(match.group(1)!);
}

bool _looksLikeNativeCrash(String text) {
  final lower = text.toLowerCase();
  return lower.contains('fatal signal') ||
      lower.contains('fatal exception') ||
      lower.contains('sigabrt') ||
      lower.contains('sigsegv') ||
      lower.contains('abort message') ||
      lower.contains('std::bad_optional_access') ||
      lower.contains('native crash');
}

Object? _flatten(Object? value) {
  if (value == null || value is String || value is num || value is bool) {
    return value;
  }
  return jsonEncode(value);
}

String _csvValue(Object? value) {
  final text = (_flatten(value) ?? '').toString();
  if (!text.contains(RegExp(r'[,"\r\n]'))) return text;
  return '"${text.replaceAll('"', '""')}"';
}

String _formatMs(Object? value) {
  if (value is! num) return '-';
  if (value < 0.1) return value.toStringAsFixed(3);
  if (value < 10) return value.toStringAsFixed(2);
  return value.toStringAsFixed(1);
}

String _reportCell(Map<String, Object?>? row) {
  if (row == null) return 'MISSING';
  final status = row['status']?.toString() ?? 'unknown';
  final p50 = (row['sync_timing'] as Map?)?['p50_ms'];
  if (status == 'ok') {
    final accuracy = row['accuracy_pass'] == true ? 'OK' : 'ACC FAIL';
    return '$accuracy ${_formatMs(p50)} ms';
  }
  if (status == 'unsupported' || status == 'unsupported_dynamic_shape') {
    return 'UNSUPPORTED';
  }
  if (status == 'not_executed_after_native_crash') return 'NOT RUN (CRASH)';
  if (status == 'native_crash') return 'NATIVE CRASH';
  final code = row['litert_status_code'] ?? row['tflite_status_code'];
  return code == null ? status.toUpperCase() : '${status.toUpperCase()} $code';
}

String _markdownReport(
  List<Map<String, Object?>> rows,
  Map<String, Object?> summary,
  List<Map<String, Object?>> shardMeta,
) {
  final byCell = <String, Map<String, Object?>>{
    for (final row in rows)
      _cellKey(
        '${row['repository']}/${row['model_name']}',
        row['mode'].toString(),
      ): row,
  };
  final buffer = StringBuffer()
    ..writeln('# Android physical-device model matrix')
    ..writeln()
    ..writeln(
      'Generated ${DateTime.now().toUtc().toIso8601String()} from five '
      'Firebase Test Lab Galaxy S23 shards.',
    )
    ..writeln()
    ..writeln(
      '- Rows: ${summary['actual_rows']}/${summary['expected_rows']} '
      '(rectangular: ${summary['rectangular']})',
    )
    ..writeln('- Status: `${jsonEncode(summary['status_counts'])}`')
    ..writeln('- Accuracy failures: ${summary['accuracy_failures']}')
    ..writeln('- Native crashes: ${summary['native_crashes']}')
    ..writeln('- Synthesized rows: ${summary['synthesized_rows']}')
    ..writeln()
    ..writeln(
      'Each successful cell is `OK p50 ms` or `ACC FAIL p50 ms`. Timing is '
      'synchronous invocation with managed I/O for CompiledModel and '
      'invoke-only for Interpreter.',
    )
    ..writeln()
    ..write('| model |');
  for (final mode in androidMatrixModes) {
    buffer.write(' ${mode.label} |');
  }
  buffer
    ..writeln()
    ..write('|---|');
  for (var i = 0; i < androidMatrixModes.length; i++) {
    buffer.write('---|');
  }
  buffer.writeln();
  for (final model in androidMatrixModels) {
    buffer.write('| ${model.label} |');
    for (final mode in androidMatrixModes) {
      buffer.write(
        ' ${_reportCell(byCell[_cellKey(model.label, mode.label)])} |',
      );
    }
    buffer.writeln();
  }
  buffer
    ..writeln()
    ..writeln('## Device metadata')
    ..writeln()
    ..writeln('```json')
    ..writeln(const JsonEncoder.withIndent('  ').convert(shardMeta))
    ..writeln('```');
  return buffer.toString();
}

Future<void> main(List<String> args) async {
  final inputRoot = Directory(_requiredOption(args, '--input-root')).absolute;
  final outputDirectory = Directory(
    _requiredOption(args, '--output-dir'),
  ).absolute;
  final sourceRun = _optionalOption(args, '--source-run') ?? 'unknown';
  if (!inputRoot.existsSync()) {
    _usage('Input root does not exist: ${inputRoot.path}.');
  }
  outputDirectory.createSync(recursive: true);

  final recordPattern = RegExp(
    r'MATRIX_RECORD kind=([a-z_]+) id=([A-Za-z0-9_-]+) '
    r'part=(\d+)/(\d+) data=([A-Za-z0-9+/=]+)',
  );
  final phasePattern = RegExp(
    r'MATRIX_PHASE model=([^ ]+) mode=([^ ]+) phase=([^ ]+)',
  );
  final records = <String, _RecordChunks>{};
  final lastPhases = <String, String>{};
  final crashByShard = <int, bool>{};
  final files =
      inputRoot
          .listSync(recursive: true, followLinks: false)
          .whereType<File>()
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  for (final file in files) {
    final text = utf8.decode(file.readAsBytesSync(), allowMalformed: true);
    final shard = _shardFromPath(file.path);
    if (shard != null && _looksLikeNativeCrash(text)) {
      crashByShard[shard] = true;
    }
    for (final match in recordPattern.allMatches(text)) {
      final kind = match.group(1)!;
      final id = match.group(2)!;
      final part = int.parse(match.group(3)!);
      final total = int.parse(match.group(4)!);
      final data = match.group(5)!;
      final key = '$kind|$id';
      final chunks = records.putIfAbsent(key, () => _RecordChunks(total));
      chunks.add(part, total, data);
    }
    for (final match in phasePattern.allMatches(text)) {
      lastPhases[_cellKey(match.group(1)!, match.group(2)!)] = match.group(3)!;
    }
  }

  final decodedByKind = <String, List<Map<String, Object?>>>{};
  final incompleteRecords = <String>[];
  for (final entry in records.entries) {
    if (!entry.value.complete) {
      incompleteRecords.add(entry.key);
      continue;
    }
    try {
      final bytes = base64.decode(entry.value.join());
      final value = (jsonDecode(utf8.decode(bytes)) as Map)
          .cast<String, Object?>();
      final kind = entry.key.substring(0, entry.key.indexOf('|'));
      decodedByKind.putIfAbsent(kind, () => []).add(value);
    } catch (error) {
      incompleteRecords.add('${entry.key}: decode failed: $error');
    }
  }

  Map<String, Map<String, Object?>> uniqueBy(
    String kind,
    String Function(Map<String, Object?>) key,
  ) {
    final result = <String, Map<String, Object?>>{};
    for (final value in decodedByKind[kind] ?? const []) {
      result[key(value)] = value;
    }
    return result;
  }

  final rowByCell = uniqueBy(
    'row',
    (row) => _cellKey(
      '${row['repository']}/${row['model_name']}',
      row['mode'].toString(),
    ),
  );
  final inventoryByModel = uniqueBy(
    'inventory',
    (item) => '${item['repository']}/${item['model_name']}',
  );
  final referenceByModel = uniqueBy(
    'reference',
    (item) => '${item['repository']}/${item['model_name']}',
  );
  final metaByShard = uniqueBy(
    'meta',
    (item) => item['matrix_shard'].toString(),
  );
  final planByShard = uniqueBy('plan', (item) => item['shard'].toString());
  final shardSummaryByShard = uniqueBy(
    'summary',
    (item) => item['shard'].toString(),
  );
  final observedRowCount = rowByCell.length;

  // A run may deliberately narrow the mode set, for example dropping the
  // Qualcomm NPU modes on a non-Qualcomm SoC where they can only fail. Only
  // modes the shards actually selected are expected, so a focused run stays
  // rectangular instead of accruing phantom "missing" cells for modes nobody
  // asked for. An unfiltered run still expects every mode.
  final requestedModeFilter = metaByShard.values
      .map((meta) => meta['mode_filter']?.toString() ?? '')
      .where((filter) => filter.isNotEmpty)
      .expand((filter) => filter.split(','))
      .map((label) => label.trim())
      .where((label) => label.isNotEmpty)
      .toSet();
  final expectedModes = requestedModeFilter.isEmpty
      ? androidMatrixModes
      : androidMatrixModes
            .where((mode) => requestedModeFilter.contains(mode.label))
            .toList();

  // The same reasoning applies to a narrowed model set. A targeted run over a
  // handful of models, which is how a per-process effect is distinguished from
  // a per-model one, must not be reported as 20-odd missing cells.
  final requestedModelFilter = metaByShard.values
      .map((meta) => meta['model_filter']?.toString() ?? '')
      .where((filter) => filter.isNotEmpty)
      .expand((filter) => filter.split(','))
      .map((label) => label.trim())
      .where((label) => label.isNotEmpty)
      .toSet();
  final expectedModels = requestedModelFilter.isEmpty
      ? androidMatrixModels
      : androidMatrixModels
            .where(
              (model) =>
                  requestedModelFilter.contains(model.label) ||
                  requestedModelFilter.contains(model.name),
            )
            .toList();

  for (final model in expectedModels) {
    for (final mode in expectedModes) {
      final key = _cellKey(model.label, mode.label);
      if (rowByCell.containsKey(key)) continue;
      final lastPhase = lastPhases[key];
      // In a single-execution run every model shares one process, so a crash
      // marker there applies to all of them regardless of their manifest bin.
      final shardCrashed =
          crashByShard[model.shard] == true ||
          crashByShard[androidMatrixAllShards] == true;
      final status = lastPhase != null && shardCrashed
          ? 'native_crash'
          : shardCrashed
          ? 'not_executed_after_native_crash'
          : 'missing_result';
      rowByCell[key] = <String, Object?>{
        ...model.toJson(),
        ...mode.toJson(),
        'model_path': model.assetPath,
        'model_bytes': inventoryByModel[model.label]?['model_bytes'] ?? 0,
        'model_sha256': inventoryByModel[model.label]?['model_sha256'] ?? '',
        'status': status,
        'phase': lastPhase ?? 'not_started',
        'error_type': 'incomplete_testlab_cell',
        'error': shardCrashed
            ? lastPhase == null
                  ? 'The shard process crashed before this cell started.'
                  : 'The native process crashed during phase $lastPhase; Dart '
                        'could not catch a process-level termination.'
            : 'No complete MATRIX_RECORD row was found in downloaded Test Lab '
                  'artifacts.',
        'synthesized': true,
      };
    }
  }

  final modelOrder = {
    for (var i = 0; i < androidMatrixModels.length; i++)
      androidMatrixModels[i].label: i,
  };
  final modeOrder = {
    for (var i = 0; i < androidMatrixModes.length; i++)
      androidMatrixModes[i].label: i,
  };
  final rows = rowByCell.values.toList()
    ..sort((a, b) {
      final aModel = '${a['repository']}/${a['model_name']}';
      final bModel = '${b['repository']}/${b['model_name']}';
      final modelComparison = (modelOrder[aModel] ?? 999).compareTo(
        modelOrder[bModel] ?? 999,
      );
      if (modelComparison != 0) return modelComparison;
      return (modeOrder[a['mode']] ?? 999).compareTo(
        modeOrder[b['mode']] ?? 999,
      );
    });

  final statusCounts = <String, int>{};
  for (final row in rows) {
    final status = row['status']?.toString() ?? 'missing_status';
    statusCounts[status] = (statusCounts[status] ?? 0) + 1;
  }
  final expectedRows = expectedModels.length * expectedModes.length;
  final accuracyFailures = rows.where(
    (row) => row['status'] == 'ok' && row['accuracy_pass'] != true,
  );
  final summary = <String, Object?>{
    // A single-execution run reports one shard, not the five-bin default.
    'expected_shards':
        metaByShard.keys.contains(androidMatrixAllShards.toString())
        ? 1
        : androidMatrixShardCount,
    'expected_modes': expectedModes.map((mode) => mode.label).toList(),
    'expected_models': expectedModels.map((model) => model.label).toList(),
    'plans_found': planByShard.length,
    'metadata_records_found': metaByShard.length,
    'shard_summaries_found': shardSummaryByShard.length,
    'expected_rows': expectedRows,
    'actual_rows': rows.length,
    'observed_rows': observedRowCount,
    'unique_rows': rowByCell.length,
    'rectangular': rows.length == expectedRows,
    'status_counts': statusCounts,
    'accuracy_failures': accuracyFailures.length,
    'successful_accuracy_checks': rows
        .where((row) => row['status'] == 'ok' && row['accuracy_pass'] == true)
        .length,
    'native_crashes': rows
        .where((row) => row['status'] == 'native_crash')
        .length,
    'synthesized_rows': rows.where((row) => row['synthesized'] == true).length,
    'incomplete_framed_records': incompleteRecords.length,
    'quality_gate_pass':
        rows.length == expectedRows &&
        accuracyFailures.isEmpty &&
        rows.every((row) => row['status'] == 'ok'),
  };
  final shardMeta = metaByShard.values.toList()
    ..sort(
      (a, b) => (a['matrix_shard'] as num).compareTo(b['matrix_shard'] as num),
    );
  final inventory = inventoryByModel.values.toList()
    ..sort(
      (a, b) => (modelOrder['${a['repository']}/${a['model_name']}'] ?? 999)
          .compareTo(
            modelOrder['${b['repository']}/${b['model_name']}'] ?? 999,
          ),
    );
  final references = referenceByModel.values.toList()
    ..sort(
      (a, b) => (modelOrder['${a['repository']}/${a['model_name']}'] ?? 999)
          .compareTo(
            modelOrder['${b['repository']}/${b['model_name']}'] ?? 999,
          ),
    );

  final result = <String, Object?>{
    'meta': {
      'schema_version': 1,
      'generated_utc': DateTime.now().toUtc().toIso8601String(),
      'source_run': sourceRun,
      'source_artifact_root': inputRoot.path,
      'repository_commits': androidMatrixRepositoryCommits,
      'input_file_count': files.length,
      'incomplete_records': incompleteRecords,
      'shards_with_native_crash_markers': [
        for (final entry in crashByShard.entries)
          if (entry.value) entry.key,
      ]..sort(),
    },
    'shard_metadata': shardMeta,
    'shard_summaries': shardSummaryByShard.values.toList(),
    'inventory': inventory,
    'references': references,
    'rows': rows,
    'summary': summary,
  };

  final jsonFile = File(
    '${outputDirectory.path}/ANDROID_MODEL_MATRIX_RESULTS.json',
  );
  jsonFile.writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert(result)}\n',
  );

  const csvColumns = <String>[
    'shard',
    'repository',
    'repository_commit',
    'model_name',
    'model_file',
    'model_bytes',
    'model_sha256',
    'engine',
    'mode',
    'delegate',
    'accelerators',
    'precision',
    'status',
    'phase',
    'error_type',
    'error',
    'litert_status_code',
    'litert_status_name',
    'tflite_status_code',
    'tflite_status_name',
    'delegate_active',
    'fully_accelerated',
    'effective_accelerators',
    'accuracy_pass',
    'accuracy_cases_passed',
    'accuracy_cases_total',
    'worst_absolute_error',
    'worst_relative_error',
    'worst_tolerance_ratio',
    'compile_ms',
    'first_inference_ms',
    'sync_timing',
    'async_status',
    'first_async_inference_ms',
    'async_timing',
    'synthesized',
  ];
  final csvFile = File(
    '${outputDirectory.path}/ANDROID_MODEL_MATRIX_RESULTS.csv',
  );
  final csv = StringBuffer()..writeln(csvColumns.join(','));
  for (final row in rows) {
    csv.writeln(csvColumns.map((column) => _csvValue(row[column])).join(','));
  }
  csvFile.writeAsStringSync(csv.toString());

  final reportFile = File(
    '${outputDirectory.path}/ANDROID_MODEL_MATRIX_REPORT.md',
  );
  reportFile.writeAsStringSync(_markdownReport(rows, summary, shardMeta));

  stdout.writeln(const JsonEncoder.withIndent('  ').convert(summary));
  stdout.writeln('JSON: ${jsonFile.path}');
  stdout.writeln('CSV: ${csvFile.path}');
  stdout.writeln('Report: ${reportFile.path}');

  if (files.isEmpty || rows.length != expectedRows) exitCode = 1;
}
