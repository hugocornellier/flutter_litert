// ignore_for_file: avoid_print

// Does repeated CompiledModel GPU compilation degrade inside one process?
//
// A Pixel 9 Pro failed LiteRtLockTensorBuffer after roughly 19 model loads in a
// single process and passed the same models in a shorter run, so there is a
// per-process leak somewhere in the GPU path. This is the flutter_litert half
// of the comparison that decides where.
//
// Google's `ai-edge-litert` Python API was clean across 300 compiles on this
// same Mac and the same Metal accelerator
// (test/benchmark/python_litert_crosscheck.py's sibling probe). This test does
// the identical thing through these bindings:
//
//   degrades here but not in Python -> the defect is in flutter_litert
//   clean here too                  -> Metal is simply unaffected, and the
//                                      Mali question stays open
//
// Deliberately mirrors the Python probe: same 18 models, same strict {gpu} at
// fp32, same compile/run/close cycle, same 300-compile target. Any divergence
// in those choices would make the comparison meaningless.
//
// Run with:
//   flutter test integration_test/macos_gpu_leak_probe_test.dart -d macos

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const _repositoriesRoot = String.fromEnvironment(
  'MODEL_REPOS_ROOT',
  defaultValue: '/Users/hugocornellier/IdeaProjects',
);

const _targetCompiles = int.fromEnvironment(
  'LEAK_PROBE_COMPILES',
  defaultValue: 300,
);

// The models that compile on strict GPU, so a failure here means decay rather
// than one of the 11 graphs known to be incompatible on every vendor.
const _models = <List<String>>[
  ['face_detection_tflite', 'face_detection_front.tflite'],
  ['face_detection_tflite', 'face_detection_back.tflite'],
  ['face_detection_tflite', 'face_detection_full_range.tflite'],
  ['face_detection_tflite', 'face_detection_short_range.tflite'],
  ['face_detection_tflite', 'face_landmark.tflite'],
  ['face_detection_tflite', 'iris_landmark.tflite'],
  ['face_detection_tflite', 'selfie_segmenter.tflite'],
  ['face_detection_tflite', 'selfie_segmenter_landscape.tflite'],
  ['face_detection_tflite', 'selfie_multiclass.tflite'],
  ['pose_detection', 'pose_landmark_full.tflite'],
  ['pose_detection', 'pose_landmark_lite.tflite'],
  ['pose_detection', 'pose_landmark_heavy.tflite'],
  ['pose_detection', 'yolov8n_float32.tflite'],
  ['hand_detection', 'hand_detection.tflite'],
  ['hand_detection', 'hand_landmark_full.tflite'],
  ['hand_detection', 'canned_gesture_classifier.tflite'],
  ['object_detection', 'efficientdet_lite0.tflite'],
  ['object_detection', 'efficientdet_lite2.tflite'],
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('repeated strict-GPU compilation does not degrade', (_) async {
    expect(Platform.isMacOS, isTrue, reason: 'Metal comparison is macOS-only.');

    var compiles = 0;
    var failures = 0;
    int? firstFailureAt;
    String? firstFailure;

    while (compiles < _targetCompiles) {
      for (final entry in _models) {
        if (compiles >= _targetCompiles) break;
        final path = '$_repositoriesRoot/${entry[0]}/assets/models/${entry[1]}';
        if (!File(path).existsSync()) {
          fail('Published model is missing: $path');
        }
        compiles++;
        CompiledModel? model;
        try {
          model = CompiledModel.fromFile(
            path,
            accelerators: const {Accelerator.gpu},
            precision: Precision.fp32,
          );
          final inputs = [
            for (final size in model.inputByteSizes)
              Float32List(size ~/ Float32List.bytesPerElement)
                ..fillRange(0, size ~/ Float32List.bytesPerElement, 0.5),
          ];
          model.run(inputs);
        } catch (error) {
          failures++;
          if (firstFailureAt == null) {
            firstFailureAt = compiles;
            firstFailure = error.toString();
            print('>>> FIRST FAILURE at compile #$compiles (${entry[1]})');
            print('>>> $error');
          }
        } finally {
          model?.close();
        }
        if (compiles % 25 == 0) {
          print('... $compiles compiles, $failures failures');
        }
      }
    }

    print('=== total compiles: $compiles, failures: $failures');
    if (firstFailureAt != null) {
      print('=== first failure at #$firstFailureAt: $firstFailure');
      print('=== flutter_litert degrades where Python did not.');
    } else {
      print('=== no degradation across $compiles compiles in one process.');
    }

    expect(
      failures,
      0,
      reason:
          'Strict-GPU compilation degraded after $firstFailureAt compiles in a '
          'single process. Python was clean at $_targetCompiles on this same '
          'Metal accelerator, so this points at flutter_litert rather than '
          'upstream LiteRT.',
    );
  });
}
