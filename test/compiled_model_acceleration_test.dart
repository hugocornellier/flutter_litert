@TestOn('mac-os || linux || windows')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Covers [CompiledModel.isFullyAccelerated].
///
/// The binding is the point here: `LiteRtCompiledModelIsFullyAccelerated` is
/// exported by both the shipped macOS dylib and the shipped Linux .so, so it is
/// looked up unconditionally and a missing symbol would fail at construction.
///
/// The *value* is deliberately not asserted per-configuration. It answers "did
/// the whole graph get accelerated", which is false for partially delegated
/// models even when the accelerator genuinely ran, and it depends on the host's
/// GPU. What is asserted is that it answers at all, stays consistent, and does
/// not throw.
void main() {
  const model = 'example/assets/simple_model.tflite';

  test('answers without throwing on a CPU-only model', () {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    expect(() => cm.isFullyAccelerated, returnsNormally);
    // Reading it must not perturb the model.
    expect(cm.run([Float32List(cm.inputByteSizes.first ~/ 4)]), isNotEmpty);
  });

  test('is stable across repeated reads', () {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    final first = cm.isFullyAccelerated;
    expect(cm.isFullyAccelerated, first);
    expect(cm.isFullyAccelerated, first);
  });

  test('remains readable after inference', () {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    final before = cm.isFullyAccelerated;
    cm.run([Float32List(cm.inputByteSizes.first ~/ 4)]);
    expect(cm.isFullyAccelerated, before);
  });

  test('throws once the model is closed', () {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    cm.close();

    expect(() => cm.isFullyAccelerated, throwsA(isA<StateError>()));
  });

  test('a GPU request still answers, whatever the host provides', () {
    final CompiledModel cm;
    try {
      cm = CompiledModel.fromBuffer(
        File(model).readAsBytesSync(),
        accelerators: {Accelerator.gpu, Accelerator.cpu},
      );
    } catch (e) {
      markTestSkipped('GPU compilation unavailable here: $e');
      return;
    }
    addTearDown(cm.close);

    expect(() => cm.isFullyAccelerated, returnsNormally);
  });
}
