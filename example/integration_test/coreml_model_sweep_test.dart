// ignore_for_file: avoid_print

// Direct-Interpreter delegate sweep across candidate models.
//
// Loads each .tflite with a bare flutter_litert Interpreter (no detector
// pipeline, no preprocessing), then times invoke() alone under
// xnnpack / gpu / coreml. This isolates the model from app-level noise so we
// can see which architectures actually benefit from CoreML/ANE on this
// (M-series) Mac.
//
//   flutter test integration_test/coreml_model_sweep_test.dart -d macos

import 'dart:io';
import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

const int iterations = 30;
const int warmup = 8;

// model label -> absolute path
const models = <String, String>{
  'species_classifier_f16':
      '/Users/hugocornellier/IdeaProjects/animal_detection/assets/models/species_classifier_float16.tflite',
  'superanimal_rtmpose_f16':
      '/Users/hugocornellier/IdeaProjects/animal_detection/assets/models/superanimal_rtmpose_s_float16.tflite',
  'selfie_multiclass':
      '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/selfie_multiclass.tflite',
  'selfie_segmenter':
      '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/selfie_segmenter.tflite',
  'mobilefacenet':
      '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/mobilefacenet.tflite',
};

const modes = <String, PerformanceConfig>{
  'xnnpack': PerformanceConfig.xnnpack(),
  'gpu': PerformanceConfig.gpu(),
  'coreml': PerformanceConfig.coreml(),
};

double _p50(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.50).floor()].toDouble();
}

double _mean(List<int> t) => t.reduce((a, b) => a + b) / t.length;

double _std(List<int> t) {
  final m = _mean(t);
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

(double, double, double) _bench(String path, PerformanceConfig cfg) {
  final bytes = File(path).readAsBytesSync();
  final (options, delegate) = InterpreterFactory.create(cfg);
  final interp = Interpreter.fromBuffer(bytes, options: options);
  interp.allocateTensors();

  // Inputs are left at their allocated (zero) state; timing of invoke() does
  // not depend on input values, only on shapes/ops, which is what we measure.
  for (int i = 0; i < warmup; i++) {
    interp.invoke();
  }
  final timings = <int>[];
  for (int i = 0; i < iterations; i++) {
    final sw = Stopwatch()..start();
    interp.invoke();
    sw.stop();
    timings.add(sw.elapsedMicroseconds);
  }

  interp.close();
  delegate?.delete();

  // microseconds -> milliseconds
  final ms = timings.map((u) => (u / 1000).round()).toList();
  return (_p50(ms), _std(ms), _mean(ms));
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final summary = <String>[];

  group('CoreML model sweep (direct Interpreter, -d macos)', () {
    models.forEach((modelName, path) {
      test(modelName, timeout: const Timeout(Duration(minutes: 10)), () async {
        final inShape = () {
          final bytes = File(path).readAsBytesSync();
          final (o, d) = InterpreterFactory.create(
            const PerformanceConfig(mode: PerformanceMode.disabled),
          );
          final ip = Interpreter.fromBuffer(bytes, options: o);
          ip.allocateTensors();
          final s = ip.getInputTensor(0).shape.toString();
          ip.close();
          d?.delete();
          return s;
        }();

        final parts = <String>[];
        modes.forEach((modeName, cfg) {
          final (p50, std, mean) = _bench(path, cfg);
          parts.add(
            '$modeName p50=${p50.toStringAsFixed(1).padLeft(6)}ms '
            'mean=${mean.toStringAsFixed(1).padLeft(6)}ms '
            'std=${std.toStringAsFixed(1).padLeft(5)}ms',
          );
        });
        final line =
            '${modelName.padRight(24)} in=$inShape\n    ${parts.join('\n    ')}';
        summary.add(line);
        print('\n>>> $line\n');
      });
    });

    tearDownAll(() {
      print('\n${'=' * 78}');
      print(
        'COREML MODEL SWEEP SUMMARY (direct Interpreter, -d macos, M-series)',
      );
      print('=' * 78);
      for (final l in summary) {
        print(l);
      }
      print('=' * 78);
    });
  });
}
