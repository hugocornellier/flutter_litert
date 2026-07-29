/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:typed_data';

import '../../native.dart';

/// Outcome of checking a [CompiledModel] against a bare-CPU [Interpreter].
///
/// [agrees] is the only field most callers need. The numbers are exposed so a
/// caller can log how far off a rejected backend was, which is the difference
/// between "wrong by 40% of the output range" and "wrong in the last decimal".
class BackendVerification {
  const BackendVerification({
    required this.agrees,
    required this.absoluteDeviation,
    required this.outputRange,
    required this.relativeDeviation,
    this.skippedReason,
    this.error,
  });

  /// Whether CompiledModel matched the reference within the tolerance.
  ///
  /// False when it disagreed, threw, or produced a differently-shaped result.
  /// Also false when verification could not run, so a caller that ignores
  /// [skipped] never treats "unknown" as "safe".
  final bool agrees;

  /// Largest absolute difference between the two backends' outputs.
  final double absoluteDeviation;

  /// Spread (max minus min) of the reference output.
  final double outputRange;

  /// [absoluteDeviation] as a fraction of [outputRange].
  ///
  /// Relative because an absolute threshold is meaningless across models whose
  /// outputs are logits, probabilities and pixel coordinates.
  final double relativeDeviation;

  /// Why verification could not run, or null if it ran.
  final String? skippedReason;

  /// The error CompiledModel threw, if it threw.
  final Object? error;

  /// Whether verification could not be performed at all.
  bool get skipped => skippedReason != null;

  @override
  String toString() {
    if (skipped) return 'BackendVerification(skipped: $skippedReason)';
    if (error != null) return 'BackendVerification(threw: $error)';
    return 'BackendVerification(agrees: $agrees, '
        'deviation: ${(relativeDeviation * 100).toStringAsFixed(3)}% of range, '
        'absolute: $absoluteDeviation, range: $outputRange)';
  }
}

/// Default tolerance: 1% of the reference output's range.
///
/// Measured separation on macOS arm64 across five shipping models is wide
/// enough that this is not a tuned knob. Healthy configurations deviated by at
/// most 0.068% (including fp16 on GPU), while known-corrupt ones deviated by
/// 42% or more. 1% sits roughly 15x above the worst honest result and 40x below
/// the mildest corruption.
const double kDefaultBackendTolerance = 0.01;

/// Checks whether [compiled] computes the same thing as a bare-CPU
/// [Interpreter] built from the same [modelBytes].
///
/// Motivation: LiteRT Next has shipped defects where CompiledModel returns
/// `kLiteRtStatusOk` while producing output that is wrong, or never written at
/// all. Neither is detectable from a status code or from timing, so the only
/// reliable check is to compare against a backend already known to be correct.
/// Run this once, at initialization, before trusting a CompiledModel.
///
/// The reference deliberately uses [PerformanceConfig.disabled] rather than
/// XNNPACK or GPU: a delegate can silently decline to run any ops, so the
/// slowest path is the only one that is unambiguously the plain CPU kernels.
///
/// This consumes one inference on [compiled]. That is safe for a healthy model,
/// and a model that fails here should be discarded rather than reused.
///
/// Only single-input float32 models are supported; anything else returns a
/// [BackendVerification] with [BackendVerification.skipped] set, and
/// [BackendVerification.agrees] false so an unchecked model is never mistaken
/// for a verified one.
BackendVerification verifyCompiledModel(
  Uint8List modelBytes,
  CompiledModel compiled, {
  double tolerance = kDefaultBackendTolerance,
}) {
  BackendVerification skip(String reason) => BackendVerification(
    agrees: false,
    absoluteDeviation: double.nan,
    outputRange: double.nan,
    relativeDeviation: double.nan,
    skippedReason: reason,
  );

  if (compiled.inputCount != 1) {
    return skip('model has ${compiled.inputCount} inputs, expected 1');
  }

  Interpreter? reference;
  Delegate? delegate;
  final List<double> expected;
  final Float32List input;
  try {
    final (options, d) = InterpreterFactory.create(PerformanceConfig.disabled);
    delegate = d;
    reference = Interpreter.fromBuffer(modelBytes, options: options);
    reference.allocateTensors();

    final inputTensors = reference.getInputTensors();
    if (inputTensors.length != 1) {
      return skip('reference has ${inputTensors.length} inputs, expected 1');
    }
    final inputFloats = _elementCount(inputTensors.first.shape);

    // Deterministic, non-degenerate ramp. A constant or all-zero input can
    // mask a backend that ignores its input entirely, and 251 being prime
    // keeps the pattern from aligning with channel or row strides.
    input = Float32List(inputFloats);
    for (var i = 0; i < inputFloats; i++) {
      input[i] = (i % 251) / 251.0;
    }

    final outputTensors = reference.getOutputTensors();
    final outputs = [
      for (final t in outputTensors) Float32List(_elementCount(t.shape)),
    ];
    reference.runForMultipleInputs(
      [input.buffer],
      {for (var i = 0; i < outputs.length; i++) i: outputs[i].buffer},
    );
    expected = [for (final o in outputs) ...o];
  } catch (e) {
    return skip('reference Interpreter failed: $e');
  } finally {
    reference?.close();
    delegate?.delete();
  }

  final List<double> actual;
  try {
    actual = [
      for (final o in compiled.run([input])) ...o,
    ];
  } catch (e) {
    return BackendVerification(
      agrees: false,
      absoluteDeviation: double.infinity,
      outputRange: _range(expected),
      relativeDeviation: double.infinity,
      error: e,
    );
  }

  if (actual.length != expected.length) {
    return BackendVerification(
      agrees: false,
      absoluteDeviation: double.infinity,
      outputRange: _range(expected),
      relativeDeviation: double.infinity,
      skippedReason:
          'output length mismatch: CompiledModel produced ${actual.length} '
          'values, reference produced ${expected.length}',
    );
  }

  var deviation = 0.0;
  for (var i = 0; i < expected.length; i++) {
    final a = actual[i];
    // NaN never compares greater, so a NaN output would otherwise slip past a
    // running maximum and read as perfect agreement.
    if (a.isNaN != expected[i].isNaN ||
        a.isInfinite != expected[i].isInfinite) {
      deviation = double.infinity;
      break;
    }
    final d = (a - expected[i]).abs();
    if (d > deviation) deviation = d;
  }

  final range = _range(expected);
  // A constant reference output has no range to normalise against, so compare
  // against the magnitude of the values themselves instead of dividing by zero.
  final scale = range > 0
      ? range
      : expected.fold<double>(0, (m, v) => v.abs() > m ? v.abs() : m);
  final relative = scale > 0
      ? deviation / scale
      : (deviation == 0 ? 0.0 : double.infinity);

  return BackendVerification(
    agrees: relative <= tolerance,
    absoluteDeviation: deviation,
    outputRange: range,
    relativeDeviation: relative,
  );
}

int _elementCount(List<int> shape) => shape.fold(1, (a, b) => a * b);

double _range(List<double> values) {
  if (values.isEmpty) return 0;
  var lo = double.infinity, hi = -double.infinity;
  for (final v in values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi - lo;
}
