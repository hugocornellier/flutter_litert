@TestOn('mac-os')
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

const _simpleModel = 'example/assets/simple_model.tflite';
const _unsupportedModel =
    'test/assets/compiled_model_npu_unsupported_abs.tflite';

void main() {
  setUp(() {
    if (Abi.current() != Abi.macosArm64) {
      markTestSkipped('The macOS NPU accelerator requires Apple Silicon.');
    }
  });

  test('works after the normal CPU environment was initialized first', () {
    final cpu = CompiledModel.fromFile(
      _simpleModel,
      accelerators: {Accelerator.cpu},
    );
    cpu.close();

    final npu = CompiledModel.fromFile(
      _simpleModel,
      accelerators: {Accelerator.npu},
    );
    addTearDown(npu.close);

    final input = Float32List(npu.inputByteSizes.first ~/ 4)..first = 3;
    expect(npu.run([input]).single.single, closeTo(7, 1e-5));
    expect(npu.accelerators, {Accelerator.npu});
    expect(npu.isFullyAccelerated, isTrue);
  });

  test('strict NPU works from bytes with managed and host buffers', () async {
    final bytes = File(_simpleModel).readAsBytesSync();
    final managed = CompiledModel.fromBuffer(
      bytes,
      accelerators: {Accelerator.npu},
    );
    final hostMemory = CompiledModel.fromBuffer(
      bytes,
      accelerators: {Accelerator.npu},
      tensorBufferMode: TensorBufferMode.hostMemory,
    );
    addTearDown(managed.close);
    addTearDown(hostMemory.close);

    final input = Float32List(managed.inputByteSizes.first ~/ 4)..first = -2;
    final expected = managed.run([input]);
    expect(await managed.runAsync([input]), expected);
    expect(hostMemory.run([input]), expected);
    expect(expected.single.single, closeTo(-3, 1e-5));
  });

  test(
    'MEAN-containing model engages NPU and stays within tolerance',
    () {
      // This model contains ten MEAN ops. It guards the required-padding patch:
      // without mutable_valid(), Core ML rejects the generated model and silently
      // leaves the whole graph on CPU.
      const path = 'example/assets/species_classifier_float16.tflite';
      final bytes = File(path).readAsBytesSync();
      final model = CompiledModel.fromBuffer(
        bytes,
        accelerators: {Accelerator.npu, Accelerator.cpu},
        precision: Precision.fp32,
      );
      addTearDown(model.close);

      final verification = verifyCompiledModel(bytes, model);
      expect(verification.skipped, isFalse, reason: verification.toString());
      expect(verification.agrees, isTrue, reason: verification.toString());
      expect(
        verification.absoluteDeviation,
        greaterThan(0),
        reason:
            'A bit-identical result would indicate that the Core ML path did '
            'not contribute to this model.',
      );
      expect(
        verification.relativeDeviation,
        lessThan(kDefaultBackendTolerance),
        reason: verification.toString(),
      );
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  test(
    'NPU + CPU agrees across representative production models',
    () {
      const models = <String>[
        'test/assets/face_detection_short_range.tflite',
        'example/assets/mobilefacenet.tflite',
        'example/assets/efficientdet_lite0.tflite',
        'example/assets/yolov8n_float32.tflite',
      ];

      for (final path in models) {
        final bytes = File(path).readAsBytesSync();
        final model = CompiledModel.fromBuffer(
          bytes,
          accelerators: {Accelerator.npu, Accelerator.cpu},
          precision: Precision.fp32,
        );
        try {
          final verification = verifyCompiledModel(bytes, model);
          expect(verification.skipped, isFalse, reason: '$path: $verification');
          expect(verification.agrees, isTrue, reason: '$path: $verification');
          expect(
            verification.absoluteDeviation,
            greaterThan(0),
            reason: '$path was bit-identical to bare CPU',
          );
        } finally {
          model.close();
        }
      }
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );

  test('rejects a zero-node mixed request instead of silently using CPU', () {
    // Compile a supported graph first so this also catches a stale diagnostics
    // counter that was not reset for the next delegate.
    CompiledModel.fromFile(
      _simpleModel,
      accelerators: {Accelerator.npu, Accelerator.cpu},
    ).close();

    final cpu = CompiledModel.fromFile(
      _unsupportedModel,
      accelerators: {Accelerator.cpu},
    );
    final input = Float32List(cpu.inputByteSizes.first ~/ 4)..first = -3;
    expect(cpu.run([input]).single.single, 3);
    cpu.close();

    expect(
      () => CompiledModel.fromFile(
        _unsupportedModel,
        accelerators: {Accelerator.npu, Accelerator.cpu},
      ),
      throwsA(
        isA<StateError>().having(
          (error) => error.message,
          'message',
          contains('delegated zero model nodes'),
        ),
      ),
    );
  });

  test('strict NPU rejects a partially supported graph', () {
    expect(
      () => CompiledModel.fromFile(
        'test/assets/face_detection_short_range.tflite',
        accelerators: {Accelerator.npu},
      ),
      throwsA(
        isA<StateError>().having(
          (error) => error.message,
          'message',
          contains('kLiteRtStatusErrorCompilation'),
        ),
      ),
    );
  });

  test('macOS NPU and GPU cannot be requested together', () {
    expect(
      () => CompiledModel.fromFile(
        _simpleModel,
        accelerators: {Accelerator.npu, Accelerator.gpu, Accelerator.cpu},
      ),
      throwsA(
        isA<UnsupportedError>().having(
          (error) => error.message,
          'message',
          contains('cannot be combined'),
        ),
      ),
    );
  });

  test('repeated NPU create, inference, and close cycles stay stable', () {
    for (var i = 0; i < 5; i++) {
      final model = CompiledModel.fromFile(
        _simpleModel,
        accelerators: {Accelerator.npu},
      );
      final input = Float32List(model.inputByteSizes.first ~/ 4)..first = i + 1;
      expect(model.run([input]).single.single, closeTo(2 * (i + 1) + 1, 1e-5));
      model.close();
    }
  });
}
