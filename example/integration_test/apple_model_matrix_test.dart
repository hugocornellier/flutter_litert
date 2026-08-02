// ignore_for_file: avoid_print, deprecated_member_use

// Apple (macOS + iOS) exhaustive model x backend accuracy and latency matrix.
//
// This is deliberately a model-level test. It feeds three deterministic,
// non-degenerate float inputs through a plain CPU Interpreter reference and
// every acceleration mode, then compares every output tensor. It therefore
// detects backend regressions independently of task-specific preprocessing.
// Labeled-image quality metrics are a complementary, later layer; the
// `accuracy_kind` emitted here is `cpu_reference_tensor_parity`.
//
// Both platforms run the identical 15-mode set so their columns are directly
// comparable. Only the model byte source differs: macOS reads the sibling
// published checkouts, iOS reads models staged into the app bundle.
//
// Run in profile mode through:
//   test/benchmark/run_apple_model_matrix.sh            (macOS)
//   test/benchmark/run_apple_model_matrix.sh --ios      (tethered iPhone)

// The host-side driver writes both JSON (full detail) and CSV (one row per
// model x mode). A backend failure never aborts the sweep: its row records the
// phase, exception type, and message, keeping the result matrix rectangular.

import 'dart:ffi';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:crypto/crypto.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

var _repositoriesRoot = const String.fromEnvironment(
  'MODEL_REPOS_ROOT',
  defaultValue: '/Users/hugocornellier/IdeaProjects',
);
var _iterations = const int.fromEnvironment('MATRIX_ITERS', defaultValue: 15);
var _warmup = const int.fromEnvironment('MATRIX_WARMUP', defaultValue: 5);
var _enforceAccuracy = const bool.fromEnvironment(
  'MATRIX_ENFORCE_ACCURACY',
  defaultValue: true,
);
var _modelFilter = const String.fromEnvironment('MATRIX_MODEL_FILTER');
var _modeFilter = const String.fromEnvironment('MATRIX_MODE_FILTER');
const _runtimeConfigPath = String.fromEnvironment('MATRIX_CONFIG_PATH');

// macOS reads the per-shard config from a host file, which lets the host build
// once and vary only the file's contents between shards. A sandboxed iPhone
// cannot read the host's disk at all, so there the same config is baked in as
// a build-time define and each shard rebuilds. It is base64 so that the JSON's
// quotes and commas survive --dart-define parsing intact.
const _runtimeConfigBase64 = String.fromEnvironment('MATRIX_CONFIG_B64');

// y ~= x when |y-x| <= absTolerance + relativeTolerance * outputScale.
// This is intentionally the same order of magnitude as
// kDefaultBackendTolerance, while the absolute floor avoids rejecting a
// harmless last-bit difference in a near-zero tensor.
const _absoluteTolerance = 1e-4;
const _relativeTolerance = 0.01;

const _requiredFixtureCount = 3;
const _fixtureCandidateNames = <String>[
  'constant_0_5',
  'ramp_0_05_0_95',
  'scrambled_0_1_0_9',
  'reverse_ramp_0_05_0_95',
  'scrambled_0_2_0_8',
];

void _loadRuntimeConfig() {
  final String source;
  final String rawConfig;
  if (_runtimeConfigBase64.isNotEmpty) {
    source = 'build-time define';
    rawConfig = utf8.decode(base64Decode(_runtimeConfigBase64));
  } else if (_runtimeConfigPath.isNotEmpty) {
    source = _runtimeConfigPath;
    final file = File(_runtimeConfigPath);
    if (!file.existsSync()) {
      throw StateError('Matrix runtime config is missing: $_runtimeConfigPath');
    }
    rawConfig = file.readAsStringSync();
  } else {
    print('>>> matrix runtime config: compile-time values only');
    return;
  }
  final config = (jsonDecode(rawConfig) as Map).cast<String, dynamic>();
  _repositoriesRoot =
      config['model_repositories_root']?.toString() ?? _repositoriesRoot;
  _modelFilter = config['model_filter']?.toString() ?? _modelFilter;
  _modeFilter = config['mode_filter']?.toString() ?? _modeFilter;
  _iterations = (config['iterations'] as num?)?.toInt() ?? _iterations;
  _warmup = (config['warmup'] as num?)?.toInt() ?? _warmup;
  _enforceAccuracy = config['enforce_accuracy'] as bool? ?? _enforceAccuracy;
  print(
    '>>> matrix runtime config: $source '
    'models=$_modelFilter modes=$_modeFilter '
    'iterations=$_iterations warmup=$_warmup',
  );
}

// macOS runs against the developer's sibling published checkouts, so it reads
// the .tflite files straight off the host filesystem and records their real
// paths as provenance. An iPhone is sandboxed and cannot reach those checkouts,
// so the same models are staged into the app bundle instead. Only the byte
// source differs; every downstream stage sees identical bytes, which is what
// keeps the macOS and iOS columns comparable.
bool get _usesBundledModels => Platform.isIOS;

class _ModelAsset {
  const _ModelAsset(this.repository, this.fileName);

  final String repository;
  final String fileName;

  String get path => _usesBundledModels
      ? assetPath
      : '$_repositoriesRoot/$repository/assets/models/$fileName';
  // Flat, with the repository folded into the file name: Flutter's
  // `assets/models/model_matrix/` pubspec entry is not recursive, and this
  // matches the layout the Android matrix already stages.
  String get assetPath => 'assets/models/model_matrix/${repository}__$fileName';
  String get name => fileName.substring(0, fileName.length - '.tflite'.length);
  String get label => '$repository/$name';
}

/// Reads [model] from whichever source this platform stages it in.
///
/// Throws with the resolved path when the model is absent, so the caller can
/// record one `model_missing` row per mode instead of aborting the matrix.
Future<Uint8List> _readModelBytes(_ModelAsset model) async {
  if (_usesBundledModels) {
    final data = await rootBundle.load(model.assetPath);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }
  final file = File(model.path);
  if (!file.existsSync()) {
    throw FileSystemException('published model is missing', model.path);
  }
  return file.readAsBytesSync();
}

// Explicit rather than directory-discovered so a missing or accidentally
// renamed published model becomes a visible model_missing row instead of
// silently shrinking the matrix.
const _models = <_ModelAsset>[
  _ModelAsset('face_detection_tflite', 'face_blendshapes.tflite'),
  _ModelAsset('face_detection_tflite', 'face_detection_back.tflite'),
  _ModelAsset('face_detection_tflite', 'face_detection_front.tflite'),
  _ModelAsset('face_detection_tflite', 'face_detection_full_range.tflite'),
  _ModelAsset(
    'face_detection_tflite',
    'face_detection_full_range_sparse.tflite',
  ),
  _ModelAsset('face_detection_tflite', 'face_detection_short_range.tflite'),
  _ModelAsset('face_detection_tflite', 'face_landmark.tflite'),
  _ModelAsset('face_detection_tflite', 'iris_landmark.tflite'),
  _ModelAsset('face_detection_tflite', 'mobilefacenet.tflite'),
  _ModelAsset('face_detection_tflite', 'selfie_multiclass.tflite'),
  _ModelAsset('face_detection_tflite', 'selfie_segmenter.tflite'),
  _ModelAsset('face_detection_tflite', 'selfie_segmenter_landscape.tflite'),
  _ModelAsset('pose_detection', 'pose_landmark_full.tflite'),
  _ModelAsset('pose_detection', 'pose_landmark_heavy.tflite'),
  _ModelAsset('pose_detection', 'pose_landmark_lite.tflite'),
  _ModelAsset('pose_detection', 'yolov8n_float32.tflite'),
  _ModelAsset('hand_detection', 'canned_gesture_classifier.tflite'),
  _ModelAsset('hand_detection', 'gesture_embedder.tflite'),
  _ModelAsset('hand_detection', 'hand_detection.tflite'),
  _ModelAsset('hand_detection', 'hand_landmark_full.tflite'),
  _ModelAsset('animal_detection', 'species_classifier_float16.tflite'),
  _ModelAsset('animal_detection', 'superanimal_rtmpose_s_float16.tflite'),
  _ModelAsset('animal_detection', 'superanimal_ssdlite_float16.tflite'),
  _ModelAsset('cat_detection', 'cat_face_landmarks_full.tflite'),
  _ModelAsset('cat_detection', 'cat_face_localizer.tflite'),
  _ModelAsset('dog_detection', 'dog_face_landmarks_full.tflite'),
  _ModelAsset('dog_detection', 'dog_face_localizer.tflite'),
  _ModelAsset('object_detection', 'efficientdet_lite0.tflite'),
  _ModelAsset('object_detection', 'efficientdet_lite2.tflite'),
];

class _TensorSpec {
  const _TensorSpec({
    required this.index,
    required this.name,
    required this.type,
    required this.shape,
    required this.bytes,
  });

  factory _TensorSpec.fromTensor(int index, Tensor tensor) => _TensorSpec(
    index: index,
    name: tensor.name,
    type: tensor.type.name,
    shape: List<int>.of(tensor.shape),
    bytes: tensor.numBytes(),
  );

  final int index;
  final String name;
  final String type;
  final List<int> shape;
  final int bytes;

  Map<String, Object?> toJson() => {
    'index': index,
    'name': name,
    'type': type,
    'shape': shape,
    'bytes': bytes,
  };
}

class _TensorSummary {
  const _TensorSummary({
    required this.values,
    required this.finiteValues,
    required this.nonFiniteValues,
    required this.min,
    required this.max,
    required this.mean,
    required this.rms,
    required this.weightedChecksum,
    required this.topIndex,
    required this.topValue,
  });

  factory _TensorSummary.from(Float32List tensor) {
    var finite = 0;
    var nonFinite = 0;
    var minValue = double.infinity;
    var maxValue = -double.infinity;
    var sum = 0.0;
    var sumSquares = 0.0;
    var checksum = 0.0;
    var topIndex = -1;
    var topValue = -double.infinity;
    for (var i = 0; i < tensor.length; i++) {
      final value = tensor[i];
      if (!value.isFinite) {
        nonFinite++;
        continue;
      }
      finite++;
      if (value < minValue) minValue = value;
      if (value > maxValue) {
        maxValue = value;
        topIndex = i;
        topValue = value;
      }
      sum += value;
      sumSquares += value * value;
      checksum += value * (1 + (i % 97));
    }
    return _TensorSummary(
      values: tensor.length,
      finiteValues: finite,
      nonFiniteValues: nonFinite,
      min: finite == 0 ? null : minValue,
      max: finite == 0 ? null : maxValue,
      mean: finite == 0 ? null : sum / finite,
      rms: finite == 0 ? null : sqrt(sumSquares / finite),
      weightedChecksum: finite == 0 ? null : checksum,
      topIndex: topIndex,
      topValue: topIndex < 0 ? null : topValue,
    );
  }

  final int values;
  final int finiteValues;
  final int nonFiniteValues;
  final double? min;
  final double? max;
  final double? mean;
  final double? rms;
  final double? weightedChecksum;
  final int topIndex;
  final double? topValue;

  Map<String, Object?> toJson() => {
    'values': values,
    'finite_values': finiteValues,
    'non_finite_values': nonFiniteValues,
    'min': min,
    'max': max,
    'mean': mean,
    'rms': rms,
    'weighted_checksum': weightedChecksum,
    'top_index': topIndex,
    'top_value': topValue,
  };
}

class _FixtureReference {
  const _FixtureReference(this.name, this.inputs, this.outputs);

  final String name;
  final List<Float32List> inputs;
  final List<Float32List> outputs;

  Map<String, Object?> toJson() => {
    'name': name,
    'input_values': inputs.fold<int>(0, (sum, input) => sum + input.length),
    'outputs': [
      for (var i = 0; i < outputs.length; i++)
        {'index': i, ..._TensorSummary.from(outputs[i]).toJson()},
    ],
  };
}

class _ModelReference {
  const _ModelReference({
    required this.inputs,
    required this.outputs,
    required this.fixtures,
    required this.rejectedFixtures,
  });

  final List<_TensorSpec> inputs;
  final List<_TensorSpec> outputs;
  final List<_FixtureReference> fixtures;
  final List<Map<String, Object?>> rejectedFixtures;

  Map<String, Object?> toJson() => {
    'status': 'ok',
    'backend': 'interpreter_cpu_4t',
    'inputs': inputs.map((spec) => spec.toJson()).toList(),
    'outputs': outputs.map((spec) => spec.toJson()).toList(),
    'fixtures': fixtures.map((fixture) => fixture.toJson()).toList(),
    'rejected_fixtures': rejectedFixtures,
  };
}

Float32List _makeInput(int values, int tensorIndex, int fixtureIndex) {
  final result = Float32List(values);
  switch (fixtureIndex) {
    case 0:
      result.fillRange(0, result.length, 0.5);
      return result;
    case 1:
      for (var i = 0; i < values; i++) {
        result[i] = 0.05 + 0.9 * ((i + tensorIndex * 17) % 251) / 250.0;
      }
      return result;
    case 2:
      for (var i = 0; i < values; i++) {
        final pattern = (i * 73 + tensorIndex * 31) % 251;
        result[i] = 0.1 + 0.8 * pattern / 250.0;
      }
      return result;
    case 3:
      for (var i = 0; i < values; i++) {
        result[i] = 0.95 - 0.9 * ((i + tensorIndex * 19) % 251) / 250.0;
      }
      return result;
    case 4:
      for (var i = 0; i < values; i++) {
        final pattern = (i * 101 + tensorIndex * 47) % 251;
        result[i] = 0.2 + 0.6 * pattern / 250.0;
      }
      return result;
    default:
      throw RangeError.index(
        fixtureIndex,
        _fixtureCandidateNames,
        'fixtureIndex',
      );
  }
}

Float32List _copyFloatTensor(Tensor tensor) {
  if (tensor.type != TensorType.float32) {
    throw UnsupportedError(
      'output[${tensor.name}] is ${tensor.type.name}; CompiledModel currently '
      'exposes Float32 outputs only',
    );
  }
  final bytes = tensor.data;
  if (bytes.lengthInBytes % Float32List.bytesPerElement != 0) {
    throw StateError(
      'output[${tensor.name}] has ${bytes.lengthInBytes} bytes, not float32 '
      'aligned',
    );
  }
  return Float32List.fromList(
    Float32List.view(
      bytes.buffer,
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ Float32List.bytesPerElement,
    ),
  );
}

_ModelReference _buildReference(Uint8List bytes) {
  final options = _newInterpreterOptions(threads: 4);
  Interpreter? interpreter;
  try {
    interpreter = Interpreter.fromBuffer(bytes, options: options);
    final inputTensors = interpreter.getInputTensors();
    final inputSpecs = <_TensorSpec>[];
    List<_TensorSpec>? outputSpecs;
    for (var i = 0; i < inputTensors.length; i++) {
      final tensor = inputTensors[i];
      if (tensor.type != TensorType.float32) {
        throw UnsupportedError(
          'input[$i] ${tensor.name} is ${tensor.type.name}; CompiledModel '
          'currently accepts Float32 inputs only',
        );
      }
      if (tensor.numBytes() == 0 ||
          tensor.numBytes() % Float32List.bytesPerElement != 0) {
        throw UnsupportedError(
          'input[$i] ${tensor.name} has unresolved/non-float byte size '
          '${tensor.numBytes()}',
        );
      }
      inputSpecs.add(_TensorSpec.fromTensor(i, tensor));
    }

    final fixtures = <_FixtureReference>[];
    final rejectedFixtures = <Map<String, Object?>>[];
    for (
      var fixtureIndex = 0;
      fixtureIndex < _fixtureCandidateNames.length &&
          fixtures.length < _requiredFixtureCount;
      fixtureIndex++
    ) {
      final inputs = <Float32List>[];
      for (var inputIndex = 0; inputIndex < inputSpecs.length; inputIndex++) {
        final spec = inputSpecs[inputIndex];
        final input = _makeInput(
          spec.bytes ~/ Float32List.bytesPerElement,
          inputIndex,
          fixtureIndex,
        );
        inputs.add(input);
      }
      final List<Float32List> outputs;
      if (outputSpecs == null) {
        // Do not obtain output handles until after the first invocation.
        // Models with dynamic arenas may relocate their TfLiteTensor structs
        // during invoke, invalidating handles obtained beforehand.
        interpreter.runInference(inputs);
        final currentOutputs = interpreter.getOutputTensors();
        outputSpecs = [
          for (var i = 0; i < currentOutputs.length; i++)
            _TensorSpec.fromTensor(i, currentOutputs[i]),
        ];
        outputs = currentOutputs.map(_copyFloatTensor).toList();
      } else {
        // This high-level path invalidates and reacquires native tensor
        // handles around every invocation. Reusing Tensor wrappers across
        // invoke is unsafe for models whose arena moves.
        outputs = [
          for (final spec in outputSpecs)
            Float32List(spec.bytes ~/ Float32List.bytesPerElement),
        ];
        interpreter.runForMultipleInputs(inputs, {
          for (var i = 0; i < outputs.length; i++) i: outputs[i],
        });
      }
      final nonFinite = outputs.fold<int>(
        0,
        (sum, output) => sum + output.where((value) => !value.isFinite).length,
      );
      if (nonFinite > 0) {
        rejectedFixtures.add({
          'name': _fixtureCandidateNames[fixtureIndex],
          'reason': 'CPU reference produced $nonFinite non-finite values',
        });
        continue;
      }
      fixtures.add(
        _FixtureReference(
          _fixtureCandidateNames[fixtureIndex],
          inputs,
          outputs,
        ),
      );
    }
    if (fixtures.length != _requiredFixtureCount) {
      throw StateError(
        'Only ${fixtures.length} finite reference fixtures could be produced; '
        'need $_requiredFixtureCount. Rejected: $rejectedFixtures',
      );
    }
    final resolvedOutputSpecs = outputSpecs;
    if (resolvedOutputSpecs == null) {
      throw StateError('CPU reference did not produce output tensor metadata.');
    }
    return _ModelReference(
      inputs: inputSpecs,
      outputs: resolvedOutputSpecs,
      fixtures: fixtures,
      rejectedFixtures: rejectedFixtures,
    );
  } finally {
    interpreter?.close();
    options.delete();
  }
}

class _AccuracyCase {
  const _AccuracyCase({
    required this.name,
    required this.passed,
    required this.outputCount,
    required this.valueCount,
    required this.maxAbsoluteError,
    required this.meanAbsoluteError,
    required this.rmse,
    required this.worstRelativeError,
    required this.worstToleranceRatio,
    required this.top1Compared,
    required this.top1Matched,
    this.reason,
  });

  final String name;
  final bool passed;
  final int outputCount;
  final int valueCount;
  final double? maxAbsoluteError;
  final double? meanAbsoluteError;
  final double? rmse;
  final double? worstRelativeError;
  final double? worstToleranceRatio;
  final int top1Compared;
  final int top1Matched;
  final String? reason;

  Map<String, Object?> toJson() => {
    'name': name,
    'passed': passed,
    'output_count': outputCount,
    'value_count': valueCount,
    'max_absolute_error': maxAbsoluteError,
    'mean_absolute_error': meanAbsoluteError,
    'rmse': rmse,
    'worst_relative_error': worstRelativeError,
    'worst_tolerance_ratio': worstToleranceRatio,
    'top1_compared': top1Compared,
    'top1_matched': top1Matched,
    'reason': reason,
  };
}

int _topIndex(Float32List values) {
  if (values.isEmpty) return -1;
  var top = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[top]) top = i;
  }
  return top;
}

_AccuracyCase _compareOutputs(
  String name,
  List<Float32List> expected,
  List<Float32List> actual,
) {
  if (expected.length != actual.length) {
    return _AccuracyCase(
      name: name,
      passed: false,
      outputCount: actual.length,
      valueCount: actual.fold<int>(0, (sum, output) => sum + output.length),
      maxAbsoluteError: null,
      meanAbsoluteError: null,
      rmse: null,
      worstRelativeError: null,
      worstToleranceRatio: null,
      top1Compared: 0,
      top1Matched: 0,
      reason:
          'output count mismatch: expected ${expected.length}, got '
          '${actual.length}',
    );
  }

  var values = 0;
  var absoluteSum = 0.0;
  var squareSum = 0.0;
  var maxAbsolute = 0.0;
  var worstRelative = 0.0;
  var worstToleranceRatio = 0.0;
  var top1Compared = 0;
  var top1Matched = 0;

  for (var outputIndex = 0; outputIndex < expected.length; outputIndex++) {
    final reference = expected[outputIndex];
    final candidate = actual[outputIndex];
    if (reference.length != candidate.length) {
      return _AccuracyCase(
        name: name,
        passed: false,
        outputCount: actual.length,
        valueCount: values,
        maxAbsoluteError: maxAbsolute,
        meanAbsoluteError: values == 0 ? null : absoluteSum / values,
        rmse: values == 0 ? null : sqrt(squareSum / values),
        worstRelativeError: worstRelative,
        worstToleranceRatio: worstToleranceRatio,
        top1Compared: top1Compared,
        top1Matched: top1Matched,
        reason:
            'output[$outputIndex] length mismatch: expected '
            '${reference.length}, got ${candidate.length}',
      );
    }

    var referenceMin = double.infinity;
    var referenceMax = -double.infinity;
    var referenceMagnitude = 0.0;
    var outputMaxError = 0.0;
    for (var i = 0; i < reference.length; i++) {
      final expectedValue = reference[i];
      final actualValue = candidate[i];
      if (!expectedValue.isFinite || !actualValue.isFinite) {
        return _AccuracyCase(
          name: name,
          passed: false,
          outputCount: actual.length,
          valueCount: values,
          maxAbsoluteError: maxAbsolute,
          meanAbsoluteError: values == 0 ? null : absoluteSum / values,
          rmse: values == 0 ? null : sqrt(squareSum / values),
          worstRelativeError: worstRelative,
          worstToleranceRatio: worstToleranceRatio,
          top1Compared: top1Compared,
          top1Matched: top1Matched,
          reason:
              'non-finite value at output[$outputIndex][$i]: '
              'reference=$expectedValue candidate=$actualValue',
        );
      }
      if (expectedValue < referenceMin) referenceMin = expectedValue;
      if (expectedValue > referenceMax) referenceMax = expectedValue;
      referenceMagnitude = max(referenceMagnitude, expectedValue.abs());
      final error = (actualValue - expectedValue).abs();
      outputMaxError = max(outputMaxError, error);
      maxAbsolute = max(maxAbsolute, error);
      absoluteSum += error;
      squareSum += error * error;
      values++;
    }

    final range = reference.isEmpty ? 0.0 : referenceMax - referenceMin;
    final scale = max(range, referenceMagnitude);
    final tolerance = _absoluteTolerance + _relativeTolerance * scale;
    final relativeError = scale == 0
        ? (outputMaxError == 0 ? 0.0 : double.infinity)
        : outputMaxError / scale;
    final toleranceRatio = tolerance == 0
        ? (outputMaxError == 0 ? 0.0 : double.infinity)
        : outputMaxError / tolerance;
    worstRelative = max(worstRelative, relativeError);
    worstToleranceRatio = max(worstToleranceRatio, toleranceRatio);

    // Top-1 is meaningful enough to record for compact vector outputs. It is
    // diagnostic, not a separate pass condition: the numeric tolerance is the
    // contract and naturally catches a material rank change.
    if (reference.length >= 2 && reference.length <= 4096) {
      top1Compared++;
      if (_topIndex(reference) == _topIndex(candidate)) top1Matched++;
    }
  }

  return _AccuracyCase(
    name: name,
    passed: worstToleranceRatio <= 1.0,
    outputCount: actual.length,
    valueCount: values,
    maxAbsoluteError: maxAbsolute,
    meanAbsoluteError: values == 0 ? 0.0 : absoluteSum / values,
    rmse: values == 0 ? 0.0 : sqrt(squareSum / values),
    worstRelativeError: worstRelative,
    worstToleranceRatio: worstToleranceRatio,
    top1Compared: top1Compared,
    top1Matched: top1Matched,
    reason: worstToleranceRatio <= 1.0
        ? null
        : 'numeric tolerance exceeded (${worstToleranceRatio.toStringAsFixed(3)}x)',
  );
}

class _TimingStats {
  const _TimingStats({
    required this.samples,
    required this.minMs,
    required this.maxMs,
    required this.meanMs,
    required this.p50Ms,
    required this.p90Ms,
    required this.stdMs,
  });

  factory _TimingStats.fromMicroseconds(List<int> microseconds) {
    final sorted = List<int>.of(microseconds)..sort();
    final meanUs = microseconds.reduce((a, b) => a + b) / microseconds.length;
    final variance =
        microseconds.fold<double>(
          0,
          (sum, value) => sum + pow(value - meanUs, 2),
        ) /
        microseconds.length;
    int percentile(double p) =>
        sorted[((sorted.length - 1) * p).round().clamp(0, sorted.length - 1)];
    return _TimingStats(
      samples: microseconds.length,
      minMs: sorted.first / 1000.0,
      maxMs: sorted.last / 1000.0,
      meanMs: meanUs / 1000.0,
      p50Ms: percentile(0.50) / 1000.0,
      p90Ms: percentile(0.90) / 1000.0,
      stdMs: sqrt(variance) / 1000.0,
    );
  }

  final int samples;
  final double minMs;
  final double maxMs;
  final double meanMs;
  final double p50Ms;
  final double p90Ms;
  final double stdMs;

  Map<String, Object?> toJson() => {
    'samples': samples,
    'min_ms': minMs,
    'max_ms': maxMs,
    'mean_ms': meanMs,
    'p50_ms': p50Ms,
    'p90_ms': p90Ms,
    'std_ms': stdMs,
  };
}

_TimingStats _benchmarkSync(void Function() invoke) {
  for (var i = 0; i < _warmup; i++) {
    invoke();
  }
  final samples = <int>[];
  for (var i = 0; i < _iterations; i++) {
    final stopwatch = Stopwatch()..start();
    invoke();
    stopwatch.stop();
    samples.add(stopwatch.elapsedMicroseconds);
  }
  return _TimingStats.fromMicroseconds(samples);
}

Future<_TimingStats> _benchmarkAsync(Future<void> Function() invoke) async {
  for (var i = 0; i < _warmup; i++) {
    await invoke();
  }
  final samples = <int>[];
  for (var i = 0; i < _iterations; i++) {
    final stopwatch = Stopwatch()..start();
    await invoke();
    stopwatch.stop();
    samples.add(stopwatch.elapsedMicroseconds);
  }
  return _TimingStats.fromMicroseconds(samples);
}

class _InterpResources {
  _InterpResources(this.options, this.delegate, [this.extraCleanup = const []]);

  final InterpreterOptions options;
  final Delegate? delegate;
  final List<void Function()> extraCleanup;

  void close() {
    try {
      delegate?.delete();
    } finally {
      for (final cleanup in extraCleanup.reversed) {
        cleanup();
      }
      options.delete();
    }
  }
}

typedef _InterpBuilder = Future<_InterpResources> Function();

InterpreterOptions _newInterpreterOptions({int? threads, Delegate? delegate}) {
  final options = InterpreterOptions()..addMediaPipeCustomOps();
  if (threads != null) options.threads = threads;
  if (delegate != null) options.addDelegate(delegate);
  return options;
}

class _InterpMode {
  const _InterpMode({
    required this.label,
    required this.delegate,
    required this.requiresActiveDelegate,
    required this.builder,
  });

  final String label;
  final String delegate;
  final bool requiresActiveDelegate;
  final _InterpBuilder builder;
}

final _interpModes = <_InterpMode>[
  _InterpMode(
    label: 'interpreter_cpu_4t',
    delegate: 'none',
    requiresActiveDelegate: false,
    builder: () async =>
        _InterpResources(_newInterpreterOptions(threads: 4), null),
  ),
  _InterpMode(
    label: 'interpreter_xnnpack_4t',
    delegate: 'xnnpack',
    requiresActiveDelegate: true,
    builder: () async {
      final delegateOptions = XNNPackDelegateOptions(numThreads: 4);
      final delegate = XNNPackDelegate(options: delegateOptions);
      final options = _newInterpreterOptions(threads: 4, delegate: delegate);
      return _InterpResources(options, delegate, [delegateOptions.delete]);
    },
  ),
  _InterpMode(
    label: 'interpreter_metal_fp16',
    delegate: 'metal',
    requiresActiveDelegate: true,
    builder: () async {
      final delegateOptions = GpuDelegateOptions(allowPrecisionLoss: true);
      final delegate = GpuDelegate(options: delegateOptions);
      final options = _newInterpreterOptions(delegate: delegate);
      return _InterpResources(options, delegate, [delegateOptions.delete]);
    },
  ),
  _InterpMode(
    label: 'interpreter_metal_fp32',
    delegate: 'metal',
    requiresActiveDelegate: true,
    builder: () async {
      final delegateOptions = GpuDelegateOptions(allowPrecisionLoss: false);
      final delegate = GpuDelegate(options: delegateOptions);
      final options = _newInterpreterOptions(delegate: delegate);
      return _InterpResources(options, delegate, [delegateOptions.delete]);
    },
  ),
  _InterpMode(
    label: 'interpreter_coreml_all',
    delegate: 'coreml',
    requiresActiveDelegate: true,
    builder: () async {
      final delegateOptions = CoreMlDelegateOptions(enabledDevices: 1);
      final delegate = CoreMlDelegate(options: delegateOptions);
      final options = _newInterpreterOptions(delegate: delegate);
      return _InterpResources(options, delegate, [delegateOptions.delete]);
    },
  ),
  _InterpMode(
    label: 'interpreter_gpu_v2',
    delegate: 'gpu_v2_gl_cl',
    requiresActiveDelegate: true,
    builder: () async {
      final delegate = GpuDelegateV2();
      final options = _newInterpreterOptions(delegate: delegate);
      return _InterpResources(options, delegate);
    },
  ),
  _InterpMode(
    label: 'interpreter_flex',
    delegate: 'flex_select_tf_ops',
    requiresActiveDelegate: true,
    builder: () async {
      if (!FlexDelegate.isAvailable) {
        throw UnsupportedError(
          'optional flutter_litert_flex delegate is not bundled in the matrix host',
        );
      }
      final delegate = await FlexDelegate.create();
      final options = _newInterpreterOptions(delegate: delegate);
      return _InterpResources(options, delegate);
    },
  ),
];

class _CmMode {
  const _CmMode(this.label, this.accelerators, this.precision);

  final String label;
  final Set<Accelerator> accelerators;
  final Precision precision;

  String get acceleratorLabel =>
      (accelerators.toList()..sort((a, b) => a.index.compareTo(b.index)))
          .map((accelerator) => accelerator.name)
          .join('+');
}

const _cmModes = <_CmMode>[
  _CmMode('compiled_cpu_fp32', {Accelerator.cpu}, Precision.fp32),
  _CmMode('compiled_gpu_fp16', {Accelerator.gpu}, Precision.fp16),
  _CmMode('compiled_gpu_fp32', {Accelerator.gpu}, Precision.fp32),
  _CmMode('compiled_npu_fp32', {Accelerator.npu}, Precision.fp32),
  _CmMode('compiled_gpu_cpu_fp32', {
    Accelerator.gpu,
    Accelerator.cpu,
  }, Precision.fp32),
  _CmMode('compiled_npu_cpu_fp32', {
    Accelerator.npu,
    Accelerator.cpu,
  }, Precision.fp32),
  _CmMode('compiled_npu_gpu_fp32', {
    Accelerator.npu,
    Accelerator.gpu,
  }, Precision.fp32),
  _CmMode('compiled_npu_gpu_cpu_fp32', {
    Accelerator.npu,
    Accelerator.gpu,
    Accelerator.cpu,
  }, Precision.fp32),
];

bool _filterIncludes(String filter, String value) =>
    filter.isEmpty || filter.split(',').contains(value);

final _selectedModels = _models
    .where(
      (model) =>
          _filterIncludes(_modelFilter, model.label) ||
          (_modelFilter.isNotEmpty &&
              _filterIncludes(_modelFilter, model.name)),
    )
    .toList();
final _selectedInterpModes = _interpModes
    .where((mode) => _filterIncludes(_modeFilter, mode.label))
    .toList();
final _selectedCmModes = _cmModes
    .where((mode) => _filterIncludes(_modeFilter, mode.label))
    .toList();

String _errorStatus(Object error) {
  if (error is UnsupportedError) return 'unsupported';
  final message = error.toString().toLowerCase();
  if (message.contains('dynamic') || message.contains('shape inference')) {
    return 'unsupported_dynamic_shape';
  }
  return 'error';
}

Map<String, Object?> _nativeStatusFields(String message) {
  MapEntry<int, String>? matchStatus(String label) {
    final match = RegExp('$label=(\\d+) \\(([^)]+)\\)').firstMatch(message);
    if (match == null) return null;
    final code = int.tryParse(match.group(1)!);
    if (code == null) return null;
    return MapEntry(code, match.group(2)!);
  }

  final liteRt = matchStatus('LiteRtStatus');
  final tfLite = matchStatus('TfLiteStatus');
  return {
    if (liteRt != null) ...{
      'litert_status_code': liteRt.key,
      'litert_status_name': liteRt.value,
    },
    if (tfLite != null) ...{
      'tflite_status_code': tfLite.key,
      'tflite_status_name': tfLite.value,
    },
  };
}

Map<String, Object?> _errorFields(
  Object error,
  StackTrace stackTrace,
  String phase,
) {
  final message = error.toString();
  return {
    'status': _errorStatus(error),
    'phase': phase,
    'error_type': error.runtimeType.toString(),
    'error': message,
    'stack': stackTrace.toString().split('\n').take(8).join('\n'),
    ..._nativeStatusFields(message),
  };
}

Map<String, Object?> _accuracyFields(List<_AccuracyCase> cases) {
  final passed = cases.where((result) => result.passed).length;
  double? worstAbsolute;
  double? worstRelative;
  double? worstToleranceRatio;
  for (final result in cases) {
    final absolute = result.maxAbsoluteError;
    final relative = result.worstRelativeError;
    final ratio = result.worstToleranceRatio;
    if (absolute != null && absolute.isFinite) {
      worstAbsolute = max(worstAbsolute ?? 0, absolute);
    }
    if (relative != null && relative.isFinite) {
      worstRelative = max(worstRelative ?? 0, relative);
    }
    if (ratio != null && ratio.isFinite) {
      worstToleranceRatio = max(worstToleranceRatio ?? 0, ratio);
    }
  }
  return {
    'accuracy_kind': 'cpu_reference_tensor_parity',
    'accuracy_pass': passed == cases.length,
    'accuracy_cases_passed': passed,
    'accuracy_cases_total': cases.length,
    'worst_absolute_error': worstAbsolute,
    'worst_relative_error': worstRelative,
    'worst_tolerance_ratio': worstToleranceRatio,
    'accuracy_cases': cases.map((result) => result.toJson()).toList(),
  };
}

Map<String, Object?> _rowBase(
  _ModelAsset model,
  int modelBytes,
  String sha,
  String engine,
  String mode,
) => {
  'repository': model.repository,
  'model_name': model.name,
  'model_file': model.fileName,
  'model_path': model.path,
  'model_bytes': modelBytes,
  'model_sha256': sha,
  'engine': engine,
  'mode': mode,
  // Absolute start time of this cell. Each mode runs in its own process, so an
  // in-process counter could not order cells across the whole sweep. On a
  // passively cooled phone this makes thermal throttling visible as drift
  // through the run rather than letting it silently depress the later modes.
  'row_started_utc': DateTime.now().toUtc().toIso8601String(),
};

void _logMatrixPhase(_ModelAsset model, String mode, String phase) {
  print('>>> MATRIX_PHASE model=${model.label} mode=$mode phase=$phase');
}

void _logMatrixRow(_ModelAsset model, String mode, Map<String, Object?> row) {
  print(
    '>>> ${model.label} $mode: ${row['status']} '
    'accuracy=${row['accuracy_pass'] ?? '-'} '
    'p50=${(row['sync_timing'] as Map?)?['p50_ms'] ?? '-'}ms',
  );
  final error = row['error'];
  if (error != null) {
    final oneLine = error.toString().replaceAll(RegExp(r'\s+'), ' ').trim();
    print(
      '>>> MATRIX_ERROR model=${model.label} mode=$mode '
      'phase=${row['phase']} type=${row['error_type']} error=$oneLine',
    );
  }
}

Future<Map<String, Object?>> _runInterpreter(
  _ModelAsset model,
  Uint8List bytes,
  String sha,
  _ModelReference reference,
  _InterpMode mode,
) async {
  final row =
      _rowBase(model, bytes.lengthInBytes, sha, 'interpreter', mode.label)
        ..addAll({
          'delegate': mode.delegate,
          'accelerators': mode.delegate,
          'precision': mode.label.contains('fp16') ? 'fp16' : 'fp32',
          'buffer_mode': 'interpreter_tensor',
          'timing_scope': 'invoke_only',
        });
  _InterpResources? resources;
  Interpreter? interpreter;
  var phase = 'delegate_create';
  void enterPhase(String value) {
    phase = value;
    _logMatrixPhase(model, mode.label, value);
  }

  enterPhase(phase);
  try {
    final compileStopwatch = Stopwatch()..start();
    resources = await mode.builder();
    enterPhase('model_create_allocate');
    interpreter = Interpreter.fromBuffer(bytes, options: resources.options);
    compileStopwatch.stop();
    row['compile_ms'] = compileStopwatch.elapsedMicroseconds / 1000.0;
    row['delegate_active'] = interpreter.hasActiveDelegate;
    row['fully_accelerated'] = null;
    if (mode.requiresActiveDelegate && !interpreter.hasActiveDelegate) {
      throw UnsupportedError(
        '${mode.delegate} declined this model; CPU fallback was rejected for '
        'the strict matrix cell',
      );
    }

    final inputTensors = interpreter.getInputTensors();
    if (inputTensors.length != reference.inputs.length) {
      throw StateError(
        'input count changed from ${reference.inputs.length} to '
        '${inputTensors.length}',
      );
    }
    final outputCount = interpreter.getOutputTensors().length;
    if (outputCount != reference.outputs.length) {
      throw StateError(
        'output count changed from ${reference.outputs.length} to '
        '$outputCount',
      );
    }

    final accuracy = <_AccuracyCase>[];
    double? firstInferenceMs;
    enterPhase('accuracy');
    for (final fixture in reference.fixtures) {
      final actual = [
        for (final spec in reference.outputs)
          Float32List(spec.bytes ~/ Float32List.bytesPerElement),
      ];
      interpreter.runForMultipleInputs(fixture.inputs, {
        for (var i = 0; i < actual.length; i++) i: actual[i],
      });
      firstInferenceMs ??=
          interpreter.lastInferenceDurationMicroseconds / 1000.0;
      accuracy.add(
        _compareOutputs('sync/${fixture.name}', fixture.outputs, actual),
      );
    }

    enterPhase('benchmark');
    final timingFixture = reference.fixtures[1];
    // Populate inputs once through the handle-safe path; warmed timing below
    // remains invoke-only as declared by timing_scope.
    interpreter.runInference(timingFixture.inputs);
    final timing = _benchmarkSync(interpreter.invoke);
    row.addAll({
      'status': 'ok',
      'phase': 'complete',
      'first_inference_ms': firstInferenceMs,
      'sync_timing': timing.toJson(),
      'async_timing': null,
      ..._accuracyFields(accuracy),
    });
    _logMatrixPhase(model, mode.label, 'complete');
  } catch (error, stackTrace) {
    row.addAll(_errorFields(error, stackTrace, phase));
  } finally {
    interpreter?.close();
    resources?.close();
  }
  return row;
}

Future<Map<String, Object?>> _runCompiled(
  _ModelAsset model,
  Uint8List bytes,
  String sha,
  _ModelReference reference,
  _CmMode mode,
) async {
  final row =
      _rowBase(model, bytes.lengthInBytes, sha, 'compiled_model', mode.label)
        ..addAll({
          'delegate': null,
          'accelerators': mode.acceleratorLabel,
          'precision': mode.precision.name,
          'buffer_mode': TensorBufferMode.managed.name,
          'timing_scope': 'run_with_managed_io',
          'delegate_active': null,
        });
  CompiledModel? compiled;
  var phase = 'compile';
  void enterPhase(String value) {
    phase = value;
    _logMatrixPhase(model, mode.label, value);
  }

  enterPhase(phase);
  try {
    final compileStopwatch = Stopwatch()..start();
    compiled = CompiledModel.fromBuffer(
      bytes,
      accelerators: mode.accelerators,
      precision: mode.precision,
      tensorBufferMode: TensorBufferMode.managed,
    );
    compileStopwatch.stop();
    row['compile_ms'] = compileStopwatch.elapsedMicroseconds / 1000.0;
    row['fully_accelerated'] = compiled.isFullyAccelerated;
    row['effective_accelerators'] =
        compiled.accelerators.map((value) => value.name).toList()..sort();

    if (compiled.inputByteSizes.length != reference.inputs.length) {
      throw StateError(
        'CompiledModel input count ${compiled.inputByteSizes.length} does not '
        'match reference ${reference.inputs.length}',
      );
    }
    for (var i = 0; i < compiled.inputByteSizes.length; i++) {
      final expectedBytes = reference.inputs[i].bytes;
      if (compiled.inputByteSizes[i] != expectedBytes) {
        throw StateError(
          'CompiledModel input[$i] has ${compiled.inputByteSizes[i]} bytes; '
          'reference has $expectedBytes',
        );
      }
    }

    final accuracy = <_AccuracyCase>[];
    double? firstInferenceMs;
    enterPhase('accuracy_sync');
    for (final fixture in reference.fixtures) {
      final stopwatch = Stopwatch()..start();
      final actual = compiled.run(fixture.inputs);
      stopwatch.stop();
      firstInferenceMs ??= stopwatch.elapsedMicroseconds / 1000.0;
      accuracy.add(
        _compareOutputs('sync/${fixture.name}', fixture.outputs, actual),
      );
    }

    double? firstAsyncInferenceMs;
    enterPhase('accuracy_async');
    for (final fixture in reference.fixtures) {
      final stopwatch = Stopwatch()..start();
      final actual = await compiled.runAsync(fixture.inputs);
      stopwatch.stop();
      firstAsyncInferenceMs ??= stopwatch.elapsedMicroseconds / 1000.0;
      accuracy.add(
        _compareOutputs('async/${fixture.name}', fixture.outputs, actual),
      );
    }

    enterPhase('benchmark_sync');
    final timingInput = reference.fixtures[1].inputs;
    final syncTiming = _benchmarkSync(() => compiled!.run(timingInput));
    enterPhase('benchmark_async');
    final asyncTiming = await _benchmarkAsync(
      () async => compiled!.runAsync(timingInput),
    );
    row.addAll({
      'status': 'ok',
      'phase': 'complete',
      'first_inference_ms': firstInferenceMs,
      'first_async_inference_ms': firstAsyncInferenceMs,
      'sync_timing': syncTiming.toJson(),
      'async_timing': asyncTiming.toJson(),
      ..._accuracyFields(accuracy),
    });
    _logMatrixPhase(model, mode.label, 'complete');
  } catch (error, stackTrace) {
    row.addAll(_errorFields(error, stackTrace, phase));
  } finally {
    compiled?.close();
  }
  return row;
}

Map<String, Object?> _unavailableRow(
  _ModelAsset model,
  String engine,
  String mode,
  String status,
  String phase,
  String error,
) => {
  ..._rowBase(model, 0, '', engine, mode),
  'status': status,
  'phase': phase,
  'error_type': status,
  'error': error,
};

Future<Map<String, Object?>> _metadata(int expectedRows) async {
  var deviceModel = 'unknown';
  var deviceExtra = '';
  var physicalDevice = true;
  try {
    if (Platform.isIOS) {
      final info = await DeviceInfoPlugin().iosInfo;
      // utsname.machine is the board identifier (iPhone16,1); info.model is the
      // generic family ("iPhone"), which cannot distinguish SoC generations.
      deviceModel = info.utsname.machine;
      physicalDevice = info.isPhysicalDevice;
      deviceExtra = '${info.systemName} ${info.systemVersion}';
    } else {
      final info = await DeviceInfoPlugin().macOsInfo;
      deviceModel = info.model;
      deviceExtra = '${info.arch} ${info.osRelease}';
    }
  } catch (_) {
    // Best effort only; Platform fields below still identify the run.
  }
  return {
    'schema_version': 1,
    'timestamp_utc': DateTime.now().toUtc().toIso8601String(),
    'platform': Platform.operatingSystem,
    'platform_version': Platform.operatingSystemVersion,
    'abi': Abi.current().toString(),
    'device_model': deviceModel,
    'device_extra': deviceExtra,
    'physical_device': physicalDevice,
    'logical_processors': Platform.numberOfProcessors,
    'build_mode': kReleaseMode
        ? 'release'
        : kProfileMode
        ? 'profile'
        : 'debug',
    'flutter_litert_commit': const String.fromEnvironment(
      'LITERT_COMMIT',
      defaultValue: 'unknown',
    ),
    'interpreter_runtime_version': Interpreter.version,
    'model_source': _usesBundledModels ? 'app_bundle' : 'host_filesystem',
    'model_repositories_root': _usesBundledModels ? null : _repositoriesRoot,
    'runtime_config_path': _runtimeConfigPath,
    'model_count': _selectedModels.length,
    'interpreter_mode_count': _selectedInterpModes.length,
    'compiled_model_mode_count': _selectedCmModes.length,
    'expected_rows': expectedRows,
    'model_filter': _modelFilter,
    'mode_filter': _modeFilter,
    'iterations': _iterations,
    'warmup': _warmup,
    'accuracy_kind': 'cpu_reference_tensor_parity',
    'accuracy_fixture_candidates': _fixtureCandidateNames,
    'accuracy_fixtures_per_model': _requiredFixtureCount,
    'absolute_tolerance': _absoluteTolerance,
    'relative_tolerance': _relativeTolerance,
    'accuracy_enforced': _enforceAccuracy,
  };
}

void main() {
  _loadRuntimeConfig();
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  final rows = <Map<String, Object?>>[];
  final inventory = <Map<String, Object?>>[];
  final references = <Map<String, Object?>>[];
  final expectedRows =
      _selectedModels.length *
      (_selectedInterpModes.length + _selectedCmModes.length);

  group('Apple published-model backend matrix', () {
    test('runs on an Apple platform', () {
      expect(
        Platform.isMacOS || Platform.isIOS,
        isTrue,
        reason: 'This matrix covers macOS and iOS only.',
      );
      expect(_iterations, greaterThan(0));
      expect(_warmup, greaterThanOrEqualTo(0));
      expect(
        _selectedModels,
        isNotEmpty,
        reason: 'Model filter matched nothing.',
      );
      expect(
        _selectedInterpModes.length + _selectedCmModes.length,
        greaterThan(0),
        reason: 'Mode filter matched nothing.',
      );
    });

    for (final model in _selectedModels) {
      test(model.label, () async {
        final Uint8List bytes;
        try {
          bytes = await _readModelBytes(model);
        } catch (error) {
          final message = 'published model is missing at ${model.path}: $error';
          inventory.add({
            'repository': model.repository,
            'model_name': model.name,
            'model_file': model.fileName,
            'model_path': model.path,
            'status': 'model_missing',
            'error': message,
          });
          references.add({
            'repository': model.repository,
            'model_name': model.name,
            'status': 'model_missing',
            'error': message,
          });
          for (final mode in _selectedInterpModes) {
            rows.add(
              _unavailableRow(
                model,
                'interpreter',
                mode.label,
                'model_missing',
                'model_read',
                message,
              ),
            );
          }
          for (final mode in _selectedCmModes) {
            rows.add(
              _unavailableRow(
                model,
                'compiled_model',
                mode.label,
                'model_missing',
                'model_read',
                message,
              ),
            );
          }
          return;
        }

        final sha = sha256.convert(bytes).toString();
        inventory.add({
          'repository': model.repository,
          'model_name': model.name,
          'model_file': model.fileName,
          'model_path': model.path,
          'model_bytes': bytes.lengthInBytes,
          'model_sha256': sha,
          'status': 'ok',
        });

        final _ModelReference reference;
        try {
          _logMatrixPhase(
            model,
            _modeFilter.isEmpty ? 'all_selected_modes' : _modeFilter,
            'reference',
          );
          reference = _buildReference(bytes);
          references.add({
            'repository': model.repository,
            'model_name': model.name,
            'model_sha256': sha,
            ...reference.toJson(),
          });
          _logMatrixPhase(
            model,
            _modeFilter.isEmpty ? 'all_selected_modes' : _modeFilter,
            'reference_complete',
          );
        } catch (error, stackTrace) {
          final message = 'CPU reference failed: $error';
          references.add({
            'repository': model.repository,
            'model_name': model.name,
            'model_sha256': sha,
            ..._errorFields(error, stackTrace, 'reference'),
          });
          for (final mode in _selectedInterpModes) {
            rows.add({
              ..._rowBase(
                model,
                bytes.lengthInBytes,
                sha,
                'interpreter',
                mode.label,
              ),
              'status': 'reference_error',
              'phase': 'reference',
              'error_type': error.runtimeType.toString(),
              'error': message,
            });
          }
          for (final mode in _selectedCmModes) {
            rows.add({
              ..._rowBase(
                model,
                bytes.lengthInBytes,
                sha,
                'compiled_model',
                mode.label,
              ),
              'status': 'reference_error',
              'phase': 'reference',
              'error_type': error.runtimeType.toString(),
              'error': message,
            });
          }
          print('>>> ${model.label}: REFERENCE ERROR $error');
          return;
        }

        for (final mode in _selectedInterpModes) {
          final result = await _runInterpreter(
            model,
            bytes,
            sha,
            reference,
            mode,
          );
          rows.add(result);
          _logMatrixRow(model, mode.label, result);
        }
        for (final mode in _selectedCmModes) {
          final result = await _runCompiled(model, bytes, sha, reference, mode);
          rows.add(result);
          _logMatrixRow(model, mode.label, result);
        }
      }, timeout: const Timeout(Duration(minutes: 30)));
    }

    tearDownAll(() async {
      final meta = await _metadata(expectedRows);
      final accuracyFailures = rows
          .where((row) => row['status'] == 'ok' && row['accuracy_pass'] != true)
          .toList();
      final referenceFailures = references
          .where((reference) => reference['status'] != 'ok')
          .toList();
      final statusCounts = <String, int>{};
      for (final row in rows) {
        final status = row['status']?.toString() ?? 'missing_status';
        statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      }
      final summary = <String, Object?>{
        'expected_rows': expectedRows,
        'actual_rows': rows.length,
        'rectangular': rows.length == expectedRows,
        'status_counts': statusCounts,
        'reference_failures': referenceFailures.length,
        'accuracy_failures': accuracyFailures.length,
        'successful_accuracy_checks': rows
            .where(
              (row) => row['status'] == 'ok' && row['accuracy_pass'] == true,
            )
            .length,
      };
      binding.reportData = <String, Object?>{
        'apple_model_matrix': {
          'meta': meta,
          'inventory': inventory,
          'references': references,
          'rows': rows,
          'summary': summary,
        },
      };

      print('\n${'=' * 100}');
      print('MACOS PUBLISHED-MODEL BACKEND MATRIX');
      print(
        'models=${_selectedModels.length} rows=${rows.length}/$expectedRows',
      );
      print('status=$statusCounts');
      print(
        'reference_failures=${referenceFailures.length} '
        'accuracy_failures=${accuracyFailures.length}',
      );
      print('=' * 100);

      expect(
        rows.length,
        expectedRows,
        reason: 'Every model x backend cell must produce exactly one row.',
      );
      if (_enforceAccuracy) {
        expect(
          referenceFailures,
          isEmpty,
          reason: 'Every model needs a valid plain-CPU reference.',
        );
        expect(
          accuracyFailures,
          isEmpty,
          reason:
              'Every backend that executed successfully must agree with the '
              'CPU reference.',
        );
      }
    });
  });
}
