// ignore_for_file: avoid_print, deprecated_member_use

// Does enabling fp16 (allow_precision_loss) change the Metal GPU result?
//
// The factory ships GpuDelegate() with no options => allow_precision_loss=false
// => full fp32 on the GPU. This test compares that against fp16 and XNNPACK on
// the same models, timing invoke() only.
//
//   flutter test integration_test/gpu_fp16_test.dart -d macos

import 'dart:io';
import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

const int iterations = 30;
const int warmup = 8;

const models = <String, String>{
  'superanimal_rtmpose_f16':
      '/Users/hugocornellier/IdeaProjects/animal_detection/assets/models/superanimal_rtmpose_s_float16.tflite',
  'selfie_multiclass':
      '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/selfie_multiclass.tflite',
  'mobilefacenet':
      '/Users/hugocornellier/IdeaProjects/face_detection_tflite/assets/models/mobilefacenet.tflite',
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

// builder returns (options, delegate-to-delete-after)
(double, double, double) _bench(
  String path,
  (InterpreterOptions, Delegate?) Function() build,
) {
  final bytes = File(path).readAsBytesSync();
  final (options, delegate) = build();
  final interp = Interpreter.fromBuffer(bytes, options: options);
  interp.allocateTensors();
  for (int i = 0; i < warmup; i++) {
    interp.invoke();
  }
  final us = <int>[];
  for (int i = 0; i < iterations; i++) {
    final sw = Stopwatch()..start();
    interp.invoke();
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  interp.close();
  delegate?.delete();
  final ms = us.map((u) => (u / 1000).round()).toList();
  return (_p50(ms), _std(ms), _mean(ms));
}

(InterpreterOptions, Delegate?) _gpu({required bool fp16}) {
  final options = InterpreterOptions();
  final del = GpuDelegate(
    options: GpuDelegateOptions(allowPrecisionLoss: fp16),
  );
  options.addDelegate(del);
  return (options, del);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final builders = <String, (InterpreterOptions, Delegate?) Function()>{
    'xnnpack': () =>
        InterpreterFactory.create(const PerformanceConfig.xnnpack()),
    'gpu_fp32': () => _gpu(fp16: false),
    'gpu_fp16': () => _gpu(fp16: true),
  };

  final summary = <String>[];

  group('GPU fp16 vs fp32 (direct Interpreter, -d macos)', () {
    models.forEach((modelName, path) {
      test(modelName, timeout: const Timeout(Duration(minutes: 10)), () async {
        final parts = <String>[];
        builders.forEach((modeName, build) {
          final (p50, std, mean) = _bench(path, build);
          parts.add(
            '${modeName.padRight(9)} p50=${p50.toStringAsFixed(1).padLeft(6)}ms '
            'mean=${mean.toStringAsFixed(1).padLeft(6)}ms '
            'std=${std.toStringAsFixed(1).padLeft(5)}ms',
          );
        });
        final line = '$modelName\n    ${parts.join('\n    ')}';
        summary.add(line);
        print('\n>>> $line\n');
      });
    });

    tearDownAll(() {
      print('\n${'=' * 70}');
      print('GPU fp16 vs fp32 SUMMARY (-d macos, M-series)');
      print('=' * 70);
      for (final l in summary) {
        print(l);
      }
      print('=' * 70);
    });
  });
}
