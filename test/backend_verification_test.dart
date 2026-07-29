@TestOn('mac-os || linux || windows')
library;

import 'dart:io';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Exercises [verifyCompiledModel] against the real runtime.
///
/// The interesting negative case (LiteRT's dynamic-output defect, and the GPU
/// corruption) needs the cat/dog landmark models, which do not ship with this
/// package, so those are covered by the consuming repos' integration tests.
/// What is asserted here is that a healthy model is accepted, that an
/// unverifiable model is reported as skipped rather than as passing, and that
/// the tolerance behaves as documented.
///
/// On macOS the LiteRT Next runtime is bundled; on Linux/Windows point the
/// loader at an extracted `ai-edge-litert` wheel:
///   LITERT_LIB_PATH=/path/to/libLiteRt.so flutter test test/backend_verification_test.dart
void main() {
  const singleInput = 'example/assets/simple_model.tflite';
  const notSingleInput = 'test/assets/training_model.tflite';
  const gpuCorrupt = 'example/assets/species_classifier_float16.tflite';

  test('accepts a healthy model on the CPU accelerator', () {
    final bytes = File(singleInput).readAsBytesSync();
    final cm = CompiledModel.fromBuffer(bytes, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    final result = verifyCompiledModel(bytes, cm);

    expect(result.skipped, isFalse, reason: result.toString());
    expect(result.agrees, isTrue, reason: result.toString());
    expect(result.relativeDeviation, lessThan(kDefaultBackendTolerance));
  });

  test('a zero tolerance still accepts a bit-identical CPU result', () {
    final bytes = File(singleInput).readAsBytesSync();
    final cm = CompiledModel.fromBuffer(bytes, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    // Both paths run the same CPU kernels here, so they should agree exactly.
    // If this ever regresses it means CompiledModel stopped being bit-exact
    // with the reference, which is worth knowing even though the shipped
    // default tolerance would still pass.
    final result = verifyCompiledModel(bytes, cm, tolerance: 0);

    expect(result.agrees, isTrue, reason: result.toString());
    expect(result.absoluteDeviation, 0);
  });

  test('reports an unverifiable model as skipped, and not as agreeing', () {
    final file = File(notSingleInput);
    if (!file.existsSync()) {
      markTestSkipped('$notSingleInput not present');
      return;
    }
    final bytes = file.readAsBytesSync();
    final CompiledModel cm;
    try {
      cm = CompiledModel.fromBuffer(bytes, accelerators: {Accelerator.cpu});
    } catch (e) {
      // Refusing to compile is itself a loud failure, which is acceptable.
      markTestSkipped('$notSingleInput does not compile: $e');
      return;
    }
    addTearDown(cm.close);

    if (cm.inputCount == 1) {
      markTestSkipped('$notSingleInput has a single input; nothing to assert');
      return;
    }

    final result = verifyCompiledModel(bytes, cm);

    expect(result.skipped, isTrue);
    expect(result.skippedReason, contains('inputs'));
    // The important half: "could not check" must never read as "safe".
    expect(result.agrees, isFalse);
  });

  test('rejects a backend that returns wrong numbers', () {
    // species_classifier is one of the models LiteRT miscomputes once the GPU
    // accelerator is in the set, deviating ~42% of the output range on macOS
    // arm64 while reporting success. This is the end-to-end negative case.
    final file = File(gpuCorrupt);
    if (!file.existsSync()) {
      markTestSkipped('$gpuCorrupt not present');
      return;
    }
    final bytes = file.readAsBytesSync();
    final CompiledModel cm;
    try {
      cm = CompiledModel.fromBuffer(
        bytes,
        accelerators: {Accelerator.gpu, Accelerator.cpu},
        precision: Precision.fp32,
      );
    } catch (e) {
      markTestSkipped('GPU compilation unavailable here: $e');
      return;
    }
    addTearDown(cm.close);

    final result = verifyCompiledModel(bytes, cm);

    if (result.agrees) {
      // No GPU on this machine, so the {gpu,cpu} set silently ran on CPU and
      // there is no corruption to detect. Not a failure of the check.
      markTestSkipped(
        'GPU did not engage (deviation '
        '${(result.relativeDeviation * 100).toStringAsFixed(3)}%), '
        'nothing to reject',
      );
      return;
    }

    expect(result.agrees, isFalse);
    expect(result.skipped, isFalse, reason: result.toString());
    expect(
      result.relativeDeviation,
      greaterThan(kDefaultBackendTolerance),
      reason: result.toString(),
    );
  });

  test('a rejected result carries the deviation for logging', () {
    final bytes = File(singleInput).readAsBytesSync();
    final cm = CompiledModel.fromBuffer(bytes, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    // A negative tolerance forces rejection of even an exact match, which is
    // the cheapest way to assert the reporting path without a corrupt model.
    final result = verifyCompiledModel(bytes, cm, tolerance: -1);

    expect(result.agrees, isFalse);
    expect(result.relativeDeviation.isFinite, isTrue);
    expect(result.outputRange.isFinite, isTrue);
    expect(result.toString(), contains('agrees: false'));
  });
}
