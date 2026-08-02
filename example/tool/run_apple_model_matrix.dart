// ignore_for_file: avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

const _models = <_ModelSpec>[
  _ModelSpec('face_detection_tflite', 'face_blendshapes.tflite'),
  _ModelSpec('face_detection_tflite', 'face_detection_back.tflite'),
  _ModelSpec('face_detection_tflite', 'face_detection_front.tflite'),
  _ModelSpec('face_detection_tflite', 'face_detection_full_range.tflite'),
  _ModelSpec(
    'face_detection_tflite',
    'face_detection_full_range_sparse.tflite',
  ),
  _ModelSpec('face_detection_tflite', 'face_detection_short_range.tflite'),
  _ModelSpec('face_detection_tflite', 'face_landmark.tflite'),
  _ModelSpec('face_detection_tflite', 'iris_landmark.tflite'),
  _ModelSpec('face_detection_tflite', 'mobilefacenet.tflite'),
  _ModelSpec('face_detection_tflite', 'selfie_multiclass.tflite'),
  _ModelSpec('face_detection_tflite', 'selfie_segmenter.tflite'),
  _ModelSpec('face_detection_tflite', 'selfie_segmenter_landscape.tflite'),
  _ModelSpec('pose_detection', 'pose_landmark_full.tflite'),
  _ModelSpec('pose_detection', 'pose_landmark_heavy.tflite'),
  _ModelSpec('pose_detection', 'pose_landmark_lite.tflite'),
  _ModelSpec('pose_detection', 'yolov8n_float32.tflite'),
  _ModelSpec('hand_detection', 'canned_gesture_classifier.tflite'),
  _ModelSpec('hand_detection', 'gesture_embedder.tflite'),
  _ModelSpec('hand_detection', 'hand_detection.tflite'),
  _ModelSpec('hand_detection', 'hand_landmark_full.tflite'),
  _ModelSpec('animal_detection', 'species_classifier_float16.tflite'),
  _ModelSpec('animal_detection', 'superanimal_rtmpose_s_float16.tflite'),
  _ModelSpec('animal_detection', 'superanimal_ssdlite_float16.tflite'),
  _ModelSpec('cat_detection', 'cat_face_landmarks_full.tflite'),
  _ModelSpec('cat_detection', 'cat_face_localizer.tflite'),
  _ModelSpec('dog_detection', 'dog_face_landmarks_full.tflite'),
  _ModelSpec('dog_detection', 'dog_face_localizer.tflite'),
  _ModelSpec('object_detection', 'efficientdet_lite0.tflite'),
  _ModelSpec('object_detection', 'efficientdet_lite2.tflite'),
];

const _modes = <_ModeSpec>[
  _ModeSpec(
    label: 'interpreter_cpu_4t',
    engine: 'interpreter',
    delegate: 'none',
    accelerators: 'none',
    precision: 'fp32',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'interpreter_xnnpack_4t',
    engine: 'interpreter',
    delegate: 'xnnpack',
    accelerators: 'xnnpack',
    precision: 'fp32',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'interpreter_metal_fp16',
    engine: 'interpreter',
    delegate: 'metal',
    accelerators: 'metal',
    precision: 'fp16',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'interpreter_metal_fp32',
    engine: 'interpreter',
    delegate: 'metal',
    accelerators: 'metal',
    precision: 'fp32',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'interpreter_coreml_all',
    engine: 'interpreter',
    delegate: 'coreml',
    accelerators: 'coreml',
    precision: 'fp32',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'interpreter_gpu_v2',
    engine: 'interpreter',
    delegate: 'gpu_v2_gl_cl',
    accelerators: 'gpu_v2_gl_cl',
    precision: 'fp32',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'interpreter_flex',
    engine: 'interpreter',
    delegate: 'flex_select_tf_ops',
    accelerators: 'flex_select_tf_ops',
    precision: 'fp32',
    bufferMode: 'interpreter_tensor',
    timingScope: 'invoke_only',
  ),
  _ModeSpec(
    label: 'compiled_cpu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'cpu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_gpu_fp16',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'gpu',
    precision: 'fp16',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_gpu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'gpu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_npu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'npu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_gpu_cpu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'cpu+gpu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_npu_cpu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'cpu+npu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_npu_gpu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'gpu+npu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
  _ModeSpec(
    label: 'compiled_npu_gpu_cpu_fp32',
    engine: 'compiled_model',
    delegate: null,
    accelerators: 'cpu+gpu+npu',
    precision: 'fp32',
    bufferMode: 'managed',
    timingScope: 'run_with_managed_io',
  ),
];

const _csvColumns = <String>[
  'timestamp_utc',
  'completed_timestamp_utc',
  // Per-cell start time, so thermal drift on a passively cooled device is
  // visible in the flat table and not only in the JSON rows.
  'row_started_utc',
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
  'delegate_delegated_nodes',
  'delegate_total_nodes',
  'delegate_partitions',
  'gpu_operations',
  'cpu_operations',
  'delegate_diagnostic',
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
  'litert_status_code',
  'litert_status_name',
  'tflite_status_code',
  'tflite_status_name',
  'stack',
  'native_log_excerpt',
  'process_log_tail',
  'process_exit_code',
  'shard_id',
  'shard_log_path',
  'crash_evidence_shard_id',
  'crash_report_reused',
  'crash_report_path',
  'native_exception',
  'native_termination',
];

class _ModelSpec {
  const _ModelSpec(this.repository, this.fileName);

  final String repository;
  final String fileName;

  String get name => fileName.substring(0, fileName.length - '.tflite'.length);
  String get label => '$repository/$name';

  String path(String repositoriesRoot) =>
      _join(repositoriesRoot, '$repository/assets/models/$fileName');
}

class _ModeSpec {
  const _ModeSpec({
    required this.label,
    required this.engine,
    required this.delegate,
    required this.accelerators,
    required this.precision,
    required this.bufferMode,
    required this.timingScope,
  });

  final String label;
  final String engine;
  final String? delegate;
  final String accelerators;
  final String precision;
  final String bufferMode;
  final String timingScope;
}

class _ProcessResult {
  const _ProcessResult({
    required this.exitCode,
    required this.stdoutText,
    required this.stderrText,
    this.launchError,
    this.launchStack,
  });

  final int? exitCode;
  final String stdoutText;
  final String stderrText;
  final Object? launchError;
  final StackTrace? launchStack;

  String get tail {
    final lines = '$stdoutText\n$stderrText'.split('\n');
    return lines.skip(lines.length > 40 ? lines.length - 40 : 0).join('\n');
  }
}

class _ExecutionMarker {
  const _ExecutionMarker({
    required this.model,
    required this.mode,
    required this.phase,
  });

  final String model;
  final String? mode;
  final String? phase;
}

class _ShardResult {
  const _ShardResult({
    required this.id,
    required this.process,
    required this.validationError,
    required this.matrix,
    required this.crashReport,
    required this.jsonPath,
    required this.logPath,
    required this.logText,
    required this.lastExecution,
    required this.nativeTermination,
  });

  final String id;
  final _ProcessResult process;
  final String? validationError;
  final Map<String, dynamic>? matrix;
  final Map<String, dynamic>? crashReport;
  final String jsonPath;
  final String logPath;
  final String logText;
  final _ExecutionMarker? lastExecution;
  final Map<String, dynamic>? nativeTermination;

  bool get accepted => matrix != null && validationError == null;
  bool get hasNativeEvidence =>
      crashReport != null || nativeTermination != null;
}

/// Where a matrix run executes.
///
/// The host is always macOS because only it can build for either target. The
/// distinction is the built artifact and the device `flutter drive` installs
/// onto: macOS runs in-place and reads models from the sibling published
/// checkouts, while iOS installs onto a tethered phone whose sandbox can only
/// reach models staged into the app bundle.
enum _Target {
  macos('macos', 'macos', 'macOS'),
  ios('ios', null, 'iOS');

  const _Target(this.buildSubcommand, this.fixedDeviceId, this.displayName);

  final String buildSubcommand;
  final String? fixedDeviceId;
  final String displayName;

  bool get bundlesModels => this == _Target.ios;
}

class _MatrixRunner {
  _MatrixRunner({
    required this.exampleRoot,
    required this.repositoryRoot,
    required this.repositoriesRoot,
    required this.outputJsonPath,
    required this.outputCsvPath,
    required this.runDirectory,
    required this.runtimeConfigPath,
    required this.models,
    required this.modes,
    required this.iterations,
    required this.warmup,
    required this.commit,
    required this.failOnQuality,
    required this.target,
    required this.deviceId,
  });

  final String exampleRoot;
  final String repositoryRoot;
  final String repositoriesRoot;
  final _Target target;
  final String deviceId;
  final String outputJsonPath;
  final String outputCsvPath;
  final String runDirectory;
  final String runtimeConfigPath;
  final List<_ModelSpec> models;
  final List<_ModeSpec> modes;
  final int iterations;
  final int warmup;
  final String commit;
  final bool failOnQuality;

  final _inventory = <String, Map<String, dynamic>>{};
  final _references = <String, Map<String, dynamic>>{};
  final _rows = <String, Map<String, dynamic>>{};
  final _attempts = <Map<String, dynamic>>[];
  final _nativeEvidence = <String, Map<String, dynamic>>{};
  Map<String, dynamic>? _baseMeta;
  var _nextShard = 0;
  late final String _startedUtc = DateTime.now().toUtc().toIso8601String();

  Future<Map<String, dynamic>> run() async {
    await Directory(runDirectory).create(recursive: true);
    await _writeRuntimeConfig(models.take(1).toList(), modes.first);
    await _buildApplication();

    for (var index = 0; index < modes.length; index++) {
      final mode = modes[index];
      print(
        '\n### MODE ${index + 1}/${modes.length}: ${mode.label} '
        '(${models.length} models)',
      );
      await _runShard(mode, models, 0);
    }

    _fillHostInventory();
    _fillUnavailableReferences();
    final matrix = _assembleMatrix();
    await _writeFinalResults(matrix);
    return matrix;
  }

  Future<void> _buildApplication() async {
    print(
      '### Building the ${target.displayName} profile integration-test '
      'application once',
    );
    final logPath = _join(runDirectory, 'build.log');
    final result = await _runProcess(
      'flutter',
      [
        'build',
        target.buildSubcommand,
        '--profile',
        '--no-pub',
        '--target=integration_test/apple_model_matrix_test.dart',
        '--dart-define=MATRIX_CONFIG_PATH=$runtimeConfigPath',
        '--dart-define=LITERT_COMMIT=$commit',
      ],
      workingDirectory: exampleRoot,
      logPath: logPath,
    );
    if (result.exitCode != 0 || result.launchError != null) {
      throw StateError(
        'Profile integration-test build failed '
        '(exit=${result.exitCode}, error=${result.launchError}).\n${result.tail}',
      );
    }
  }

  Future<void> _runShard(
    _ModeSpec mode,
    List<_ModelSpec> shardModels,
    int depth,
  ) async {
    final result = await _launchShard(mode, shardModels, depth);
    if (result.accepted) {
      _mergeMatrix(result.matrix!, result);
      print(
        '### ${result.id} accepted: ${mode.label}, '
        '${shardModels.length} model(s)',
      );
      return;
    }

    final failure =
        result.validationError ??
        result.process.launchError?.toString() ??
        'flutter drive exited ${result.process.exitCode} before reportData';
    print(
      '### ${result.id} failed: ${mode.label}, '
      '${shardModels.length} model(s): $failure',
    );

    final activeLabel = result.lastExecution?.model;
    final activeIndex = activeLabel == null
        ? -1
        : shardModels.indexWhere((model) => model.label == activeLabel);
    if (shardModels.length > 1 &&
        result.hasNativeEvidence &&
        activeIndex >= 0) {
      // A crash report or a native fatal line plus the last model/phase marker
      // identifies the cell directly. Preserve that evidence and rerun only
      // the unaffected prefix/suffix instead of repeatedly rediscovering the
      // same native termination through midpoint bisection.
      if (activeIndex > 0) {
        await _runShard(mode, shardModels.sublist(0, activeIndex), depth + 1);
      }
      final model = shardModels[activeIndex];
      final row = _syntheticFailureRow(model, mode, result);
      _rows[_rowKey(row)] = row;
      print(
        '### ${result.id} directly isolated '
        '${model.label}/${mode.label}: '
        '${row['status']} ${row['error_type']}',
      );
      if (activeIndex + 1 < shardModels.length) {
        await _runShard(mode, shardModels.sublist(activeIndex + 1), depth + 1);
      }
      return;
    }

    if (shardModels.length > 1) {
      final midpoint = shardModels.length ~/ 2;
      await _runShard(mode, shardModels.sublist(0, midpoint), depth + 1);
      await _runShard(mode, shardModels.sublist(midpoint), depth + 1);
      return;
    }

    final model = shardModels.single;
    final row = _syntheticFailureRow(model, mode, result);
    _rows[_rowKey(row)] = row;
    print(
      '### ${result.id} isolated ${model.label}/${mode.label}: '
      '${row['status']} ${row['error_type']}',
    );
  }

  Future<_ShardResult> _launchShard(
    _ModeSpec mode,
    List<_ModelSpec> shardModels,
    int depth,
  ) async {
    final id = 'shard-${(++_nextShard).toString().padLeft(4, '0')}';
    final jsonPath = _join(runDirectory, '$id.json');
    final csvPath = _join(runDirectory, '$id.csv');
    final logPath = _join(runDirectory, '$id.log');
    await _writeRuntimeConfig(shardModels, mode);
    final crashFilesBefore = _crashFileSnapshot();
    final started = DateTime.now().toUtc();
    print(
      '### $id depth=$depth ${mode.label}: '
      '${shardModels.first.label}'
      '${shardModels.length == 1 ? '' : ' … ${shardModels.last.label}'}',
    );
    final environment = <String, String>{
      ...Platform.environment,
      'APPLE_MATRIX_JSON': jsonPath,
      'APPLE_MATRIX_CSV': csvPath,
      'APPLE_MATRIX_MERGE': 'false',
    };
    final process = await _runProcess(
      'flutter',
      [
        'drive',
        '--profile',
        '--no-pub',
        // macOS reuses the single prebuilt app and varies only the contents of
        // the host config file. iOS cannot read that file, so its config rides
        // in as a build-time define, which only takes effect if the shard is
        // allowed to rebuild.
        if (target != _Target.ios) '--no-build',
        '--driver=test_driver/apple_model_matrix_driver.dart',
        '--target=integration_test/apple_model_matrix_test.dart',
        '--dart-define=MATRIX_CONFIG_PATH=$runtimeConfigPath',
        if (target == _Target.ios)
          '--dart-define=MATRIX_CONFIG_B64=$_shardConfigBase64',
        '--dart-define=LITERT_COMMIT=$commit',
        '-d',
        deviceId,
        // iOS pays a per-shard rebuild and install that macOS does not, on top
        // of slower device execution, so it needs more headroom before the
        // driver gives up.
        if (target == _Target.ios) '--timeout=3600' else '--timeout=1800',
      ],
      workingDirectory: exampleRoot,
      environment: environment,
      logPath: logPath,
    );
    final logText = await File(logPath).readAsString();
    final lastExecution = _lastExecutionMarker(logText);
    final nativeTermination = _nativeTerminationFromLog(logText);

    Map<String, dynamic>? matrix;
    String? validationError;
    final jsonFile = File(jsonPath);
    if (await jsonFile.exists()) {
      try {
        matrix = (jsonDecode(await jsonFile.readAsString()) as Map)
            .cast<String, dynamic>();
        validationError = _validateShard(matrix, mode, shardModels);
      } catch (error) {
        validationError = 'invalid shard JSON: $error';
      }
    } else {
      validationError = 'result JSON was not produced';
    }

    Map<String, dynamic>? crashReport;
    if (validationError != null || process.exitCode != 0) {
      final crashFile = await _newCrashFile(
        crashFilesBefore,
        started,
        attempts: shardModels.length == 1 ? 40 : 12,
      );
      if (crashFile != null) {
        crashReport = await _parseCrashReport(crashFile);
      }
    }

    if ((crashReport != null || nativeTermination != null) &&
        lastExecution != null) {
      final evidenceKey = '${mode.label}|${lastExecution.model}';
      final evidence = <String, dynamic>{
        'source_shard_id': id,
        'phase': lastExecution.phase,
        'crash_report': crashReport,
        'native_termination': nativeTermination,
        'log_path': logPath,
        'native_log_excerpt': _modelLogExcerpt(
          logText,
          lastExecution.model,
          mode.label,
        ),
      };
      final previous = _nativeEvidence[evidenceKey];
      if (previous == null ||
          (previous['crash_report'] == null && crashReport != null)) {
        _nativeEvidence[evidenceKey] = evidence;
      }
    }

    _attempts.add({
      'id': id,
      'timestamp_utc': started.toIso8601String(),
      'depth': depth,
      'mode': mode.label,
      'models': shardModels.map((model) => model.label).toList(),
      'expected_rows': shardModels.length,
      'process_exit_code': process.exitCode,
      'launch_error': process.launchError?.toString(),
      'validation_error': validationError,
      'accepted': matrix != null && validationError == null,
      'result_json_path': jsonPath,
      'log_path': logPath,
      'crash_report_path': crashReport?['path'],
      'last_started_model': lastExecution?.model,
      'last_phase': lastExecution?.phase,
      'native_termination': nativeTermination,
    });

    return _ShardResult(
      id: id,
      process: process,
      validationError: validationError,
      matrix: matrix,
      crashReport: crashReport,
      jsonPath: jsonPath,
      logPath: logPath,
      logText: logText,
      lastExecution: lastExecution,
      nativeTermination: nativeTermination,
    );
  }

  /// Config for the shard about to run, base64 of its JSON.
  ///
  /// Only meaningful for iOS, where it is passed as a build-time define
  /// because the device cannot read [runtimeConfigPath] on the host.
  var _shardConfigBase64 = '';

  Future<void> _writeRuntimeConfig(
    List<_ModelSpec> shardModels,
    _ModeSpec mode,
  ) async {
    final config = <String, Object?>{
      'model_repositories_root': repositoriesRoot,
      'model_filter': shardModels.map((model) => model.label).join(','),
      'mode_filter': mode.label,
      'iterations': iterations,
      'warmup': warmup,
      // Collection must always finish. The host applies the requested quality
      // gate only after the complete rectangular dataset has been written.
      'enforce_accuracy': false,
    };
    final encoded = const JsonEncoder.withIndent('  ').convert(config);
    _shardConfigBase64 = base64Encode(utf8.encode(encoded));
    await _writeFile(File(runtimeConfigPath), '$encoded\n');
  }

  String? _validateShard(
    Map<String, dynamic> matrix,
    _ModeSpec mode,
    List<_ModelSpec> shardModels,
  ) {
    final rawRows = matrix['rows'];
    if (rawRows is! List) return 'dataset has no rows list';
    final actual = <String>{};
    for (final value in rawRows) {
      if (value is! Map) return 'dataset contains a non-map row';
      final row = value.cast<String, dynamic>();
      actual.add(_rowKey(row));
    }
    final expected = {
      for (final model in shardModels)
        '${model.label}|${mode.engine}|${mode.label}',
    };
    if (rawRows.length != expected.length || actual.length != expected.length) {
      return 'row cardinality mismatch: expected ${expected.length}, '
          'got ${rawRows.length} (${actual.length} unique)';
    }
    final missing = expected.difference(actual);
    final extra = actual.difference(expected);
    if (missing.isNotEmpty || extra.isNotEmpty) {
      return 'row key mismatch: missing=$missing extra=$extra';
    }
    return null;
  }

  void _mergeMatrix(Map<String, dynamic> matrix, _ShardResult shard) {
    final meta = matrix['meta'];
    if (_baseMeta == null && meta is Map) {
      _baseMeta = meta.cast<String, dynamic>();
    }
    for (final value in (matrix['inventory'] as List? ?? const [])) {
      final item = (value as Map).cast<String, dynamic>();
      _inventory[_modelKey(item)] = item;
    }
    for (final value in (matrix['references'] as List? ?? const [])) {
      final item = (value as Map).cast<String, dynamic>();
      _references.putIfAbsent(_modelKey(item), () => item);
    }
    for (final value in (matrix['rows'] as List? ?? const [])) {
      final row = (value as Map).cast<String, dynamic>();
      row['shard_id'] = shard.id;
      row['shard_log_path'] = shard.logPath;
      final excerpt = _modelLogExcerpt(
        shard.logText,
        _modelKey(row),
        row['mode']?.toString() ?? '',
      );
      row.addAll(_delegationDiagnostics(excerpt));
      if (row['status'] != 'ok') {
        if (excerpt != null && excerpt.isNotEmpty) {
          row['native_log_excerpt'] = excerpt;
        }
      }
      _rows[_rowKey(row)] = row;
    }
  }

  Map<String, dynamic> _syntheticFailureRow(
    _ModelSpec model,
    _ModeSpec mode,
    _ShardResult shard,
  ) {
    final file = File(model.path(repositoriesRoot));
    var bytes = 0;
    var digest = '';
    if (file.existsSync()) {
      final contents = file.readAsBytesSync();
      bytes = contents.lengthInBytes;
      digest = sha256.convert(contents).toString();
    }
    final evidenceKey = '${mode.label}|${model.label}';
    final cachedEvidence = _nativeEvidence[evidenceKey];
    final cachedCrash = cachedEvidence == null
        ? null
        : (cachedEvidence['crash_report'] as Map?)?.cast<String, dynamic>();
    final cachedNativeTermination = cachedEvidence == null
        ? null
        : (cachedEvidence['native_termination'] as Map?)
              ?.cast<String, dynamic>();
    final cachedSourceShardId = cachedEvidence == null
        ? null
        : cachedEvidence['source_shard_id']?.toString();
    final cachedLogExcerpt = cachedEvidence == null
        ? null
        : cachedEvidence['native_log_excerpt']?.toString();
    final cachedPhase = cachedEvidence == null
        ? null
        : cachedEvidence['phase']?.toString();
    final crash = shard.crashReport ?? cachedCrash;
    final nativeTermination =
        shard.nativeTermination ?? cachedNativeTermination;
    final crashReportReused = shard.crashReport == null && crash != null;
    final evidenceShardId = crashReportReused
        ? cachedSourceShardId
        : (shard.hasNativeEvidence ? shard.id : cachedSourceShardId);
    final exception = (crash?['exception'] as Map?)?.cast<String, dynamic>();
    final appleTermination = (crash?['termination'] as Map?)
        ?.cast<String, dynamic>();
    final crashFrames = (crash?['faulting_thread_frames'] as List?)
        ?.map((value) => value.toString())
        .toList();
    final errorType = crash != null
        ? [
            exception?['type'],
            exception?['signal'],
          ].whereType<Object>().join('/')
        : nativeTermination?['type']?.toString() ??
              shard.process.launchError?.runtimeType.toString() ??
              'ProcessExit';
    final appleTerminationText = appleTermination == null
        ? null
        : [
            appleTermination['namespace'],
            appleTermination['indicator'],
          ].whereType<Object>().join(': ');
    final message = crash != null
        ? 'Native app crash escaped Dart: $errorType'
              '${appleTerminationText == null || appleTerminationText.isEmpty ? '' : '; $appleTerminationText'}'
        : nativeTermination != null
        ? 'Native process termination escaped Dart: '
              '${nativeTermination['message'] ?? errorType}'
        : 'Integration process ended before returning a dataset row: '
              '${shard.validationError ?? shard.process.launchError ?? 'exit ${shard.process.exitCode}'}';
    final currentExcerpt = _modelLogExcerpt(
      shard.logText,
      model.label,
      mode.label,
    );
    final nativeLogExcerpt = currentExcerpt ?? cachedLogExcerpt;
    final loggedPhase = shard.lastExecution?.model == model.label
        ? shard.lastExecution?.phase
        : cachedPhase;
    return <String, dynamic>{
      'repository': model.repository,
      'model_name': model.name,
      'model_file': model.fileName,
      'model_path': model.path(repositoriesRoot),
      'model_bytes': bytes,
      'model_sha256': digest,
      'engine': mode.engine,
      'mode': mode.label,
      'delegate': mode.delegate,
      'accelerators': mode.accelerators,
      'effective_accelerators': null,
      'precision': mode.precision,
      'buffer_mode': mode.bufferMode,
      'timing_scope': mode.timingScope,
      'status': crash != null
          ? 'native_crash'
          : nativeTermination != null
          ? 'native_termination'
          : 'process_failure',
      'phase': _inferNativePhase(crashFrames, loggedPhase),
      'delegate_active': null,
      ..._delegationDiagnostics(nativeLogExcerpt),
      'fully_accelerated': null,
      'accuracy_kind': 'cpu_reference_tensor_parity',
      'accuracy_pass': null,
      'accuracy_cases_passed': 0,
      'accuracy_cases_total': 0,
      'error_type': errorType.isEmpty ? 'NativeCrash' : errorType,
      'error': message,
      'stack':
          crashFrames?.take(30).join('\n') ??
          nativeLogExcerpt ??
          shard.process.tail,
      'native_log_excerpt': nativeLogExcerpt,
      'process_log_tail': shard.process.tail,
      'process_exit_code': shard.process.exitCode,
      'crash_report_path': crash?['path'],
      'native_crash': crash,
      'native_termination': nativeTermination,
      'shard_id': shard.id,
      'shard_log_path': shard.logPath,
      'crash_evidence_shard_id': evidenceShardId,
      'crash_report_reused': crashReportReused,
    };
  }

  String _inferNativePhase(List<String>? frames, String? loggedPhase) {
    if (loggedPhase != null && loggedPhase.isNotEmpty) return loggedPhase;
    final stack = (frames ?? const []).join('\n');
    if (stack.contains('LiteRtCreateCompiledModel') ||
        stack.contains('CreateCompiledModel') ||
        stack.contains('ModifyGraphWithDelegate')) {
      return 'native_compile';
    }
    if (stack.contains('Invoke') ||
        stack.contains('RunAsync') ||
        stack.contains('Run')) {
      return 'native_inference';
    }
    return 'native_process';
  }

  void _fillHostInventory() {
    for (final model in models) {
      _inventory.putIfAbsent(model.label, () {
        final file = File(model.path(repositoriesRoot));
        if (!file.existsSync()) {
          return {
            'repository': model.repository,
            'model_name': model.name,
            'model_file': model.fileName,
            'model_path': file.path,
            'status': 'model_missing',
            'error': 'published model is missing at ${file.path}',
            'source': 'host_orchestrator',
          };
        }
        final contents = file.readAsBytesSync();
        return {
          'repository': model.repository,
          'model_name': model.name,
          'model_file': model.fileName,
          'model_path': file.path,
          'model_bytes': contents.lengthInBytes,
          'model_sha256': sha256.convert(contents).toString(),
          'status': 'ok',
          'source': 'host_orchestrator',
        };
      });
    }
  }

  void _fillUnavailableReferences() {
    for (final model in models) {
      _references.putIfAbsent(model.label, () {
        final related = _rows.values
            .where(
              (row) =>
                  row['repository'] == model.repository &&
                  row['model_name'] == model.name,
            )
            .toList();
        final crash = related.cast<Map<String, dynamic>?>().firstWhere(
          (row) =>
              row?['status'] == 'native_crash' ||
              row?['status'] == 'native_termination',
          orElse: () => null,
        );
        return {
          'repository': model.repository,
          'model_name': model.name,
          'model_sha256': _inventory[model.label]?['model_sha256'],
          'status': 'reference_unavailable',
          'phase': 'reference',
          'error_type': crash?['error_type'] ?? 'MissingReference',
          'error':
              crash?['error'] ??
              'No completed shard returned the CPU reference metadata.',
        };
      });
    }
  }

  Map<String, dynamic> _assembleMatrix() {
    final expectedKeys = {
      for (final model in models)
        for (final mode in modes) '${model.label}|${mode.engine}|${mode.label}',
    };
    final actualKeys = _rows.keys.toSet();
    final missingKeys = expectedKeys.difference(actualKeys).toList()..sort();
    final extraKeys = actualKeys.difference(expectedKeys).toList()..sort();
    if (missingKeys.isNotEmpty || extraKeys.isNotEmpty) {
      throw StateError(
        'Orchestrator coverage mismatch: missing=$missingKeys extra=$extraKeys',
      );
    }

    final rows = _rows.values.toList()
      ..sort((a, b) => _rowKey(a).compareTo(_rowKey(b)));
    final inventory = _inventory.values.toList()
      ..sort((a, b) => _modelKey(a).compareTo(_modelKey(b)));
    final references = _references.values.toList()
      ..sort((a, b) => _modelKey(a).compareTo(_modelKey(b)));
    final statusCounts = <String, int>{};
    for (final row in rows) {
      final status = row['status']?.toString() ?? 'missing_status';
      statusCounts[status] = (statusCounts[status] ?? 0) + 1;
    }
    final accuracyFailures = rows
        .where((row) => row['status'] == 'ok' && row['accuracy_pass'] != true)
        .length;
    final referenceFailures = references
        .where((reference) => reference['status'] != 'ok')
        .length;
    final executionFailures = rows.where((row) {
      final status = row['status']?.toString();
      return status != 'ok' &&
          status != 'unsupported' &&
          status != 'unsupported_dynamic_shape';
    }).length;
    final completed = DateTime.now().toUtc().toIso8601String();
    final baseMeta = _baseMeta ?? <String, dynamic>{};
    final meta = <String, dynamic>{
      ...baseMeta,
      'schema_version': 2,
      'timestamp_utc': _startedUtc,
      'completed_timestamp_utc': completed,
      'flutter_litert_commit': commit,
      'platform': baseMeta['platform'] ?? Platform.operatingSystem,
      'platform_version':
          baseMeta['platform_version'] ?? Platform.operatingSystemVersion,
      'logical_processors':
          baseMeta['logical_processors'] ?? Platform.numberOfProcessors,
      'build_mode': 'profile',
      'model_repositories_root': repositoriesRoot,
      'model_count': models.length,
      'interpreter_mode_count': modes
          .where((mode) => mode.engine == 'interpreter')
          .length,
      'compiled_model_mode_count': modes
          .where((mode) => mode.engine == 'compiled_model')
          .length,
      'expected_rows': expectedKeys.length,
      'model_filter': models.length == _models.length
          ? ''
          : models.map((model) => model.label).join(','),
      'mode_filter': modes.length == _modes.length
          ? 'process_isolated_all_modes'
          : modes.map((mode) => mode.label).join(','),
      'iterations': iterations,
      'warmup': warmup,
      'accuracy_enforced_in_app': false,
      'quality_gate_enforced_by_orchestrator': failOnQuality,
      'orchestrator': {
        'strategy': 'one_mode_per_process_recursive_model_bisection',
        'runtime_config_path': runtimeConfigPath,
        'run_directory': runDirectory,
        'attempt_count': _attempts.length,
        'accepted_shard_count': _attempts
            .where((attempt) => attempt['accepted'] == true)
            .length,
        'failed_shard_count': _attempts
            .where((attempt) => attempt['accepted'] != true)
            .length,
        'attempts': _attempts,
      },
    };
    final rectangular = rows.length == expectedKeys.length;
    final qualityPass =
        rectangular &&
        referenceFailures == 0 &&
        accuracyFailures == 0 &&
        executionFailures == 0;
    return {
      'meta': meta,
      'inventory': inventory,
      'references': references,
      'rows': rows,
      'summary': {
        'expected_rows': expectedKeys.length,
        'actual_rows': rows.length,
        'unique_rows': actualKeys.length,
        'rectangular': rectangular,
        'status_counts': statusCounts,
        'reference_failures': referenceFailures,
        'accuracy_failures': accuracyFailures,
        'execution_failures': executionFailures,
        'native_crashes':
            (statusCounts['native_crash'] ?? 0) +
            (statusCounts['native_termination'] ?? 0),
        'native_crash_reports': statusCounts['native_crash'] ?? 0,
        'native_terminations_without_report':
            statusCounts['native_termination'] ?? 0,
        'process_failures': statusCounts['process_failure'] ?? 0,
        'successful_accuracy_checks': rows
            .where(
              (row) => row['status'] == 'ok' && row['accuracy_pass'] == true,
            )
            .length,
        'quality_gate_pass': qualityPass,
      },
    };
  }

  Future<void> _writeFinalResults(Map<String, dynamic> matrix) async {
    await _writeFile(
      File(outputJsonPath),
      '${const JsonEncoder.withIndent('  ').convert(matrix)}\n',
    );
    final meta = (matrix['meta'] as Map).cast<String, dynamic>();
    final rows = (matrix['rows'] as List).map(
      (row) => (row as Map).cast<String, dynamic>(),
    );
    final csv = StringBuffer()..writeln(_csvColumns.join(','));
    for (final row in rows) {
      final flat = _flatten(meta, row);
      csv.writeln(
        _csvColumns.map((column) => _csvCell(flat[column])).join(','),
      );
    }
    await _writeFile(File(outputCsvPath), csv.toString());
    final summary = (matrix['summary'] as Map).cast<String, dynamic>();
    print('\n### COMPLETE DATASET');
    print('JSON: $outputJsonPath');
    print('CSV:  $outputCsvPath');
    print('Rows: ${summary['actual_rows']}/${summary['expected_rows']}');
    print('Status: ${summary['status_counts']}');
    print(
      'Accuracy failures: ${summary['accuracy_failures']}; '
      'native crashes: ${summary['native_crashes']}; '
      'quality gate: ${summary['quality_gate_pass']}',
    );
  }
}

_ExecutionMarker? _lastExecutionMarker(String logText) {
  final phaseMatches = RegExp(
    r'>>> MATRIX_PHASE model=([^\s]+) mode=([^\s]+) phase=([^\s]+)',
  ).allMatches(logText);
  if (phaseMatches.isNotEmpty) {
    final match = phaseMatches.last;
    return _ExecutionMarker(
      model: match.group(1)!,
      mode: match.group(2),
      phase: match.group(3),
    );
  }

  // Compatibility fallback for a host built before phase markers existed.
  final modelMatches = RegExp(
    r'macOS published-model backend matrix '
    r'([A-Za-z0-9_]+/[A-Za-z0-9_]+)',
  ).allMatches(logText);
  if (modelMatches.isEmpty) return null;
  return _ExecutionMarker(
    model: modelMatches.last.group(1)!,
    mode: null,
    phase: null,
  );
}

Map<String, dynamic>? _nativeTerminationFromLog(String logText) {
  final cppWithDetail = RegExp(
    r'libc\+\+abi:.*uncaught exception of type\s+(.+?):\s+([^\n]+)',
  ).firstMatch(logText);
  if (cppWithDetail != null) {
    final type = cppWithDetail.group(1)!.trim();
    final detail = cppWithDetail.group(2)!.trim();
    return {
      'kind': 'uncaught_cpp_exception',
      'type': type,
      'message': '$type: $detail',
    };
  }
  final cppWithoutDetail = RegExp(
    r'libc\+\+abi:.*uncaught exception of type\s+([^\s]+)',
  ).firstMatch(logText);
  if (cppWithoutDetail != null) {
    final type = cppWithoutDetail.group(1)!.trim();
    return {
      'kind': 'uncaught_cpp_exception',
      'type': type,
      'message': 'uncaught C++ exception of type $type',
    };
  }

  final nativeFatal = RegExp(
    r'^(.*(?:Segmentation fault|Bus error|Abort trap|Fatal signal|'
    r'SIGSEGV|SIGBUS|SIGABRT).*)$',
    multiLine: true,
    caseSensitive: false,
  ).allMatches(logText);
  if (nativeFatal.isEmpty) return null;
  final line = nativeFatal.last.group(1)!.trim();
  return {
    'kind': 'native_fatal_log',
    'type': 'NativeFatalSignal',
    'message': line,
  };
}

String? _modelLogExcerpt(String logText, String model, String mode) {
  final modelMarker = 'macOS published-model backend matrix $model';
  var start = logText.indexOf(modelMarker);
  if (start < 0) {
    start = logText.indexOf('>>> MATRIX_PHASE model=$model mode=$mode');
  }
  if (start < 0) return null;

  final resultMarker = '>>> $model $mode:';
  var end = logText.indexOf(resultMarker, start);
  if (end >= 0) {
    final newline = logText.indexOf('\n', end);
    end = newline < 0 ? logText.length : newline;
  } else {
    final nextModel = RegExp(
      r'macOS published-model backend matrix '
      r'[A-Za-z0-9_]+/[A-Za-z0-9_]+',
    ).firstMatch(logText.substring(start + modelMarker.length));
    end = nextModel == null
        ? logText.length
        : start + modelMarker.length + nextModel.start;
  }

  final lines = logText.substring(start, end).split('\n').where((line) {
    final trimmed = line.trimLeft();
    if (trimmed.isEmpty) return false;
    if (trimmed.startsWith('VMServiceFlutterDriver:')) return false;
    if (trimmed == 'Unhandled exception:') return false;
    if (trimmed.startsWith('DriverError:')) return false;
    if (trimmed.startsWith('Original error:')) return false;
    if (trimmed.startsWith('Original stack trace:')) return false;
    if (trimmed.startsWith('#')) return false;
    if (trimmed.startsWith('<asynchronous suspension>')) return false;
    return true;
  }).toList();
  if (lines.isEmpty) return null;
  final limited = lines.length > 80 ? lines.sublist(lines.length - 80) : lines;
  var excerpt = limited.join('\n').trim();
  const maxChars = 16000;
  if (excerpt.length > maxChars) {
    excerpt =
        '<earlier native log truncated>\n'
        '${excerpt.substring(excerpt.length - maxChars)}';
  }
  return excerpt.isEmpty ? null : excerpt;
}

Map<String, Object?> _delegationDiagnostics(String? logExcerpt) {
  if (logExcerpt == null || logExcerpt.isEmpty) return const {};
  final result = <String, Object?>{};
  final summaries = <String>[];

  final nodeMatches = RegExp(
    r'(?:TfLiteFlexDelegate|CoreML) delegate:\s*'
    r'(\d+) nodes delegated out of (\d+) nodes,?\s*'
    r'with (\d+) partitions\.?',
  ).allMatches(logExcerpt);
  if (nodeMatches.isNotEmpty) {
    final match = nodeMatches.last;
    result.addAll({
      'delegate_delegated_nodes': int.parse(match.group(1)!),
      'delegate_total_nodes': int.parse(match.group(2)!),
      'delegate_partitions': int.parse(match.group(3)!),
    });
    summaries.add(match.group(0)!);
  }

  final gpuMatches = RegExp(
    r'(\d+) operations will run on the GPU, and the remaining '
    r'(\d+) operations will run on the CPU\.?',
  ).allMatches(logExcerpt);
  if (gpuMatches.isNotEmpty) {
    final match = gpuMatches.last;
    result.addAll({
      'gpu_operations': int.parse(match.group(1)!),
      'cpu_operations': int.parse(match.group(2)!),
    });
    summaries.add(match.group(0)!);
  }

  if (summaries.isNotEmpty) {
    result['delegate_diagnostic'] = summaries.join(' | ');
  }
  return result;
}

Future<_ProcessResult> _runProcess(
  String executable,
  List<String> arguments, {
  required String workingDirectory,
  required String logPath,
  Map<String, String>? environment,
}) async {
  final stdoutBuffer = StringBuffer();
  final stderrBuffer = StringBuffer();
  final logFile = File(logPath);
  await logFile.parent.create(recursive: true);
  final log = logFile.openWrite();
  try {
    final process = await Process.start(
      executable,
      arguments,
      workingDirectory: workingDirectory,
      environment: environment,
    );
    final stdoutDone = process.stdout.transform(utf8.decoder).forEach((chunk) {
      stdout.write(chunk);
      stdoutBuffer.write(chunk);
      log.write(chunk);
    });
    final stderrDone = process.stderr.transform(utf8.decoder).forEach((chunk) {
      stderr.write(chunk);
      stderrBuffer.write(chunk);
      log.write(chunk);
    });
    final code = await process.exitCode;
    await Future.wait([stdoutDone, stderrDone]);
    await log.flush();
    return _ProcessResult(
      exitCode: code,
      stdoutText: stdoutBuffer.toString(),
      stderrText: stderrBuffer.toString(),
    );
  } catch (error, stackTrace) {
    final message = 'Failed to launch $executable: $error\n$stackTrace\n';
    stderr.write(message);
    log.write(message);
    await log.flush();
    return _ProcessResult(
      exitCode: null,
      stdoutText: stdoutBuffer.toString(),
      stderrText: stderrBuffer.toString(),
      launchError: error,
      launchStack: stackTrace,
    );
  } finally {
    await log.close();
  }
}

Set<String> _crashFileSnapshot() {
  final directory = _diagnosticReportsDirectory();
  if (directory == null || !directory.existsSync()) return const {};
  try {
    return directory
        .listSync()
        .whereType<File>()
        .where(_isMatrixCrashReport)
        .map((file) => file.path)
        .toSet();
  } catch (_) {
    return const {};
  }
}

Future<File?> _newCrashFile(
  Set<String> before,
  DateTime processStartedUtc, {
  required int attempts,
}) async {
  final directory = _diagnosticReportsDirectory();
  if (directory == null) return null;
  for (var attempt = 0; attempt < attempts; attempt++) {
    try {
      final candidates =
          directory
              .listSync()
              .whereType<File>()
              .where(_isMatrixCrashReport)
              .where((file) {
                final threshold = processStartedUtc.subtract(
                  const Duration(seconds: 2),
                );
                final crashTimestamp = _crashTimestampUtc(file);
                if (crashTimestamp != null &&
                    crashTimestamp.isBefore(threshold)) {
                  return false;
                }
                if (!before.contains(file.path)) return true;
                return file.lastModifiedSync().toUtc().isAfter(threshold);
              })
              .toList()
            ..sort(
              (a, b) => b.lastModifiedSync().compareTo(a.lastModifiedSync()),
            );
      if (candidates.isNotEmpty) return candidates.first;
    } catch (_) {
      return null;
    }
    await Future<void>.delayed(const Duration(milliseconds: 500));
  }
  return null;
}

Directory? _diagnosticReportsDirectory() {
  final userHome = Platform.environment['HOME'];
  if (userHome == null || userHome.isEmpty) return null;
  return Directory(_join(userHome, 'Library/Logs/DiagnosticReports'));
}

bool _isMatrixCrashReport(File file) {
  final name = file.uri.pathSegments.last;
  return name.startsWith('flutter_litert_example-') && name.endsWith('.ips');
}

DateTime? _crashTimestampUtc(File file) {
  try {
    final firstLine = file.readAsLinesSync().first;
    final header = (jsonDecode(firstLine) as Map).cast<String, dynamic>();
    final raw = header['timestamp']?.toString();
    if (raw == null) return null;
    final normalized = raw
        .replaceFirst(' ', 'T')
        .replaceFirstMapped(
          RegExp(r' ([+-]\d{2})(\d{2})$'),
          (match) => '${match[1]}:${match[2]}',
        );
    return DateTime.parse(normalized).toUtc();
  } catch (_) {
    return null;
  }
}

Future<Map<String, dynamic>> _parseCrashReport(File file) async {
  try {
    final text = await file.readAsString();
    final newline = text.indexOf('\n');
    if (newline < 0) {
      throw const FormatException('missing IPS header separator');
    }
    final header = (jsonDecode(text.substring(0, newline)) as Map)
        .cast<String, dynamic>();
    final body = (jsonDecode(text.substring(newline + 1)) as Map)
        .cast<String, dynamic>();
    final images = (body['usedImages'] as List? ?? const [])
        .map((value) => (value as Map).cast<String, dynamic>())
        .toList();
    final threads = body['threads'] as List? ?? const [];
    final faultingIndex = (body['faultingThread'] as num?)?.toInt();
    Map<String, dynamic>? faultingThread;
    if (faultingIndex != null &&
        faultingIndex >= 0 &&
        faultingIndex < threads.length) {
      faultingThread = (threads[faultingIndex] as Map).cast<String, dynamic>();
    } else {
      for (final value in threads) {
        final thread = (value as Map).cast<String, dynamic>();
        if (thread['triggered'] == true) {
          faultingThread = thread;
          break;
        }
      }
    }
    final frames = <String>[];
    for (final value in (faultingThread?['frames'] as List? ?? const [])) {
      final frame = (value as Map).cast<String, dynamic>();
      final imageIndex = (frame['imageIndex'] as num?)?.toInt();
      final imageName =
          imageIndex != null && imageIndex >= 0 && imageIndex < images.length
          ? images[imageIndex]['name']?.toString()
          : null;
      final symbol =
          frame['symbol']?.toString() ??
          'imageOffset=${frame['imageOffset'] ?? 'unknown'}';
      final location = frame['symbolLocation'];
      frames.add(
        '${imageName == null ? '' : '$imageName!'}$symbol'
        '${location == null ? '' : ' + $location'}',
      );
    }
    return {
      'path': file.path,
      'app_name': header['app_name'],
      'timestamp': header['timestamp'],
      'incident_id': header['incident_id'] ?? body['incident'],
      'os_version': header['os_version'],
      'model_code': body['modelCode'],
      'pid': body['pid'],
      'process_launch': body['procLaunch'],
      'capture_time': body['captureTime'],
      'exception': body['exception'],
      'termination': body['termination'],
      'application_specific_information': body['asi'],
      'faulting_thread_id': faultingThread?['id'],
      'faulting_thread_queue': faultingThread?['queue'],
      'faulting_thread_frames': frames.take(50).toList(),
    };
  } catch (error, stackTrace) {
    return {
      'path': file.path,
      'parse_error': error.toString(),
      'parse_stack': stackTrace.toString().split('\n').take(8).join('\n'),
    };
  }
}

Map<String, Object?> _flatten(
  Map<String, dynamic> meta,
  Map<String, dynamic> row,
) {
  final sync =
      (row['sync_timing'] as Map?)?.cast<String, dynamic>() ?? const {};
  final async =
      (row['async_timing'] as Map?)?.cast<String, dynamic>() ?? const {};
  final crash =
      (row['native_crash'] as Map?)?.cast<String, dynamic>() ?? const {};
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
    'native_exception': crash['exception'],
    'native_termination': crash['termination'],
  };
}

String _csvCell(Object? value) {
  if (value == null) return '';
  final text = value is List || value is Map
      ? jsonEncode(value)
      : value.toString();
  if (text.contains(',') || text.contains('"') || text.contains('\n')) {
    return '"${text.replaceAll('"', '""')}"';
  }
  return text;
}

String _modelKey(Map<String, dynamic> value) =>
    '${value['repository']}/${value['model_name']}';

String _rowKey(Map<String, dynamic> value) =>
    '${_modelKey(value)}|${value['engine']}|${value['mode']}';

String _join(String parent, String child) {
  if (parent.endsWith(Platform.pathSeparator)) return '$parent$child';
  return '$parent${Platform.pathSeparator}$child';
}

Future<void> _writeFile(File file, String contents) async {
  await file.parent.create(recursive: true);
  await file.writeAsString(contents, flush: true);
}

int _positiveInt(
  String name,
  int defaultValue, {
  bool allowZero = false,
  String? override,
}) {
  final raw = override ?? Platform.environment[name];
  if (raw == null || raw.isEmpty) return defaultValue;
  final value = int.tryParse(raw);
  if (value == null || (allowZero ? value < 0 : value <= 0)) {
    throw ArgumentError.value(raw, name, 'must be a valid integer');
  }
  return value;
}

bool _boolEnvironment(String name, bool defaultValue) {
  final raw = Platform.environment[name]?.toLowerCase();
  if (raw == null || raw.isEmpty) return defaultValue;
  if (raw == 'true' || raw == '1' || raw == 'yes') return true;
  if (raw == 'false' || raw == '0' || raw == 'no') return false;
  throw ArgumentError.value(raw, name, 'must be true or false');
}

bool _filterIncludes(String filter, String label, [String? shortName]) {
  if (filter.isEmpty) return true;
  final values = filter.split(',').map((value) => value.trim()).toSet();
  return values.contains(label) ||
      (shortName != null && values.contains(shortName));
}

String? _option(List<String> arguments, String name) {
  final prefix = '$name=';
  final matches = arguments.where((value) => value.startsWith(prefix)).toList();
  if (matches.length > 1) {
    throw ArgumentError('Option $name was provided more than once.');
  }
  return matches.isEmpty ? null : matches.single.substring(prefix.length);
}

Future<String> _commitAt(String repositoryRoot) async {
  final override = Platform.environment['LITERT_COMMIT'];
  if (override != null && override.isNotEmpty) return override;
  final result = await Process.run('git', [
    'rev-parse',
    '--short',
    'HEAD',
  ], workingDirectory: repositoryRoot);
  if (result.exitCode != 0) return 'unknown';
  return result.stdout.toString().trim();
}

/// Resolves the tethered iOS device to drive.
///
/// Auto-detection deliberately refuses when more than one iOS device is
/// attached: silently picking one would attribute a whole dataset to the wrong
/// hardware, and device identity is the point of a physical-device matrix.
Future<String> _resolveIosDeviceId(String? requested) async {
  if (requested != null) return requested;
  final result = await Process.run('flutter', ['devices', '--machine']);
  if (result.exitCode != 0) {
    throw StateError('Could not list Flutter devices: ${result.stderr}');
  }
  final devices = (jsonDecode(result.stdout as String) as List)
      .cast<Map<String, dynamic>>()
      .where((device) => device['targetPlatform'] == 'ios')
      .where((device) => device['emulator'] != true)
      .toList();
  if (devices.isEmpty) {
    throw StateError(
      'No physical iOS device is connected. Attach and unlock the iPhone, '
      'trust this Mac, then re-run.',
    );
  }
  if (devices.length > 1) {
    final names = devices.map((d) => '${d['name']} (${d['id']})').join(', ');
    throw StateError(
      'Multiple iOS devices are connected; pass --device <udid>. Found: '
      '$names',
    );
  }
  final device = devices.single;
  print('### iOS target: ${device['name']} (${device['id']})');
  return device['id'] as String;
}

Future<void> main(List<String> arguments) async {
  if (!Platform.isMacOS) {
    stderr.writeln(
      'The published-model matrix must be driven from a macOS host.',
    );
    exitCode = 64;
    return;
  }
  try {
    final exampleRoot = Directory.current.absolute.path;
    if (!File(_join(exampleRoot, 'pubspec.yaml')).existsSync()) {
      throw StateError(
        'Run this tool from the flutter_litert/example directory. '
        'Current directory: $exampleRoot',
      );
    }
    final repositoryRoot = Directory(exampleRoot).parent.path;
    final repositoriesRoot =
        _option(arguments, '--model-root') ??
        Platform.environment['MODEL_REPOS_ROOT'] ??
        Directory(repositoryRoot).parent.path;
    final modelFilter =
        _option(arguments, '--model-filter') ??
        Platform.environment['MATRIX_MODEL_FILTER'] ??
        '';
    final modeFilter =
        _option(arguments, '--mode-filter') ??
        Platform.environment['MATRIX_MODE_FILTER'] ??
        '';
    final selectedModels = _models
        .where((model) => _filterIncludes(modelFilter, model.label, model.name))
        .toList();
    final selectedModes = _modes
        .where((mode) => _filterIncludes(modeFilter, mode.label))
        .toList();
    if (selectedModels.isEmpty) {
      throw ArgumentError('MATRIX_MODEL_FILTER matched no manifest model.');
    }
    if (selectedModes.isEmpty) {
      throw ArgumentError('MATRIX_MODE_FILTER matched no matrix mode.');
    }

    final runStamp = DateTime.now().toUtc().toIso8601String().replaceAll(
      RegExp(r'[^0-9A-Za-z]'),
      '-',
    );
    final target = arguments.contains('--ios') ? _Target.ios : _Target.macos;
    final deviceId =
        target.fixedDeviceId ??
        await _resolveIosDeviceId(_option(arguments, '--device'));
    // Each target owns its own dataset. Defaulting both to the same file would
    // let an iOS run silently overwrite the recorded macOS results.
    final prefix = target == _Target.ios ? 'IOS' : 'MACOS';
    final runDirectory = _join(
      repositoryRoot,
      'build/codex-tmp/${target.buildSubcommand}-model-matrix-$runStamp-$pid',
    );
    final outputJsonPath =
        _option(arguments, '--output-json') ??
        Platform.environment['APPLE_MATRIX_JSON'] ??
        _join(
          repositoryRoot,
          'test/benchmark/${prefix}_MODEL_MATRIX_RESULTS.json',
        );
    final outputCsvPath =
        _option(arguments, '--output-csv') ??
        Platform.environment['APPLE_MATRIX_CSV'] ??
        _join(
          repositoryRoot,
          'test/benchmark/${prefix}_MODEL_MATRIX_RESULTS.csv',
        );
    final runner = _MatrixRunner(
      exampleRoot: exampleRoot,
      repositoryRoot: repositoryRoot,
      repositoriesRoot: repositoriesRoot,
      outputJsonPath: outputJsonPath,
      outputCsvPath: outputCsvPath,
      runDirectory: runDirectory,
      runtimeConfigPath: _join(runDirectory, 'runtime-config.json'),
      models: selectedModels,
      modes: selectedModes,
      iterations: _positiveInt(
        'MATRIX_ITERS',
        15,
        override: _option(arguments, '--iterations'),
      ),
      warmup: _positiveInt(
        'MATRIX_WARMUP',
        5,
        allowZero: true,
        override: _option(arguments, '--warmup'),
      ),
      commit: await _commitAt(repositoryRoot),
      failOnQuality: arguments.contains('--no-enforce')
          ? false
          : _boolEnvironment('MATRIX_ENFORCE_ACCURACY', true),
      target: target,
      deviceId: deviceId,
    );
    final matrix = await runner.run();
    final summary = (matrix['summary'] as Map).cast<String, dynamic>();
    if (runner.failOnQuality && summary['quality_gate_pass'] != true) {
      stderr.writeln(
        'The complete dataset was written, but its quality gate failed.',
      );
      exitCode = 2;
    }
  } catch (error, stackTrace) {
    stderr.writeln('macOS model matrix runner failed: $error');
    stderr.writeln(stackTrace);
    exitCode = 1;
  }
}
