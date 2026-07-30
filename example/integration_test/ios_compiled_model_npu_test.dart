import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// End-to-end checks for CompiledModel's Core ML NPU path.
///
/// These tests run in both the iOS simulator and on a physical iPhone. The
/// simulator validates packaging, registration, Core ML conversion, inference,
/// and fallback rejection. Only the physical-device run can exercise an Apple
/// Neural Engine.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final isSupportedPlatform = Platform.isIOS || Platform.isMacOS;

  Future<Uint8List> loadAsset(String path) async {
    final data = await rootBundle.load(path);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  testWidgets('strict NPU compiles, runs, and owns the whole simple graph', (
    tester,
  ) async {
    if (!isSupportedPlatform) {
      markTestSkipped('Core ML NPU validation is Apple-platform only.');
      return;
    }

    final bytes = await loadAsset('assets/simple_model.tflite');
    final model = CompiledModel.fromBuffer(
      bytes,
      accelerators: {Accelerator.npu},
      precision: Precision.fp32,
    );
    addTearDown(model.close);

    final input = Float32List(model.inputByteSizes.single ~/ 4)..first = 3;
    expect(model.run([input]).single.single, closeTo(7, 1e-4));
    expect(model.accelerators, {Accelerator.npu});
    expect(model.isFullyAccelerated, isTrue);
  });

  testWidgets(
    'NPU + CPU agrees with bare CPU across representative models',
    (tester) async {
      if (!isSupportedPlatform) {
        markTestSkipped('Core ML NPU validation is Apple-platform only.');
        return;
      }

      const assets = <String>[
        // Ten MEAN ops: guards the required Core ML pooling-padding patch.
        'assets/species_classifier_float16.tflite',
        // Face embedding / landmark-style graph.
        'assets/mobilefacenet.tflite',
        // Object detection graph.
        'assets/efficientdet_lite0.tflite',
        // YOLO detection graph.
        'assets/yolov8n_float32.tflite',
        // Production pose graph.
        'assets/pose_landmark_heavy.tflite',
      ];

      for (final asset in assets) {
        final bytes = await loadAsset(asset);
        final model = CompiledModel.fromBuffer(
          bytes,
          accelerators: {Accelerator.npu, Accelerator.cpu},
          precision: Precision.fp32,
        );
        try {
          final verification = verifyCompiledModel(bytes, model);
          expect(
            verification.skipped,
            isFalse,
            reason: '$asset: $verification',
          );
          expect(verification.agrees, isTrue, reason: '$asset: $verification');
          expect(
            verification.absoluteDeviation,
            greaterThan(0),
            reason:
                '$asset was bit-identical to bare CPU despite reported Core '
                'ML delegation',
          );
        } finally {
          model.close();
        }
      }
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );

  testWidgets('mixed mode rejects a zero-node Core ML fallback', (
    tester,
  ) async {
    if (!isSupportedPlatform) {
      markTestSkipped('Core ML NPU validation is Apple-platform only.');
      return;
    }

    // First compile a supported model so this also catches a stale native
    // delegated-node counter that was not reset for the next delegate.
    final supported = await loadAsset('assets/simple_model.tflite');
    CompiledModel.fromBuffer(
      supported,
      accelerators: {Accelerator.npu, Accelerator.cpu},
    ).close();

    final unsupported = await loadAsset(
      'assets/compiled_model_npu_unsupported_abs.tflite',
    );
    expect(
      () => CompiledModel.fromBuffer(
        unsupported,
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

  testWidgets('NPU and GPU cannot be requested together', (tester) async {
    if (!isSupportedPlatform) {
      markTestSkipped('Core ML NPU validation is Apple-platform only.');
      return;
    }

    final bytes = await loadAsset('assets/simple_model.tflite');
    expect(
      () => CompiledModel.fromBuffer(
        bytes,
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
}
