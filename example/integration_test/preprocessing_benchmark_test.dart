// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Measures the per-frame preprocessing cost of OpenCV against the pure-Dart
/// `image` package.
///
/// flutter_litert depends on neither: preprocessing belongs to the application.
/// The choice still matters, because at camera frame rates preprocessing can
/// cost more than inference, and OpenCV is a large native dependency that also
/// prevents the host app from building for the iOS simulator.
///
/// Both paths run the same pipeline the example app uses for detection, so the
/// numbers describe a real workload rather than an isolated resize:
///
///   decode JPEG -> letterbox resize -> pad to square -> normalize to Float32
///
/// Run with:
///   flutter test integration_test/preprocessing_benchmark_test.dart -d macos
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const inputSize = 320;
  const warmup = 3;
  const iterations = 20;

  testWidgets('OpenCV versus the image package on one detection frame', (
    _,
  ) async {
    final data = await rootBundle.load('assets/samples/street.jpg');
    var jpeg = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

    // The bundled sample is smaller than a real camera frame, and decode and
    // resize both scale with pixel count, so upscale to 1080p first. Otherwise
    // the absolute numbers understate what a live pipeline actually pays.
    final upscaled = img.copyResize(
      img.decodeJpg(jpeg)!,
      width: 1920,
      height: 1080,
      interpolation: img.Interpolation.linear,
    );
    jpeg = Uint8List.fromList(img.encodeJpg(upscaled, quality: 90));

    // Report the source dimensions so the numbers can be scaled to other
    // resolutions, since decode dominates and is resolution-bound.
    final probe = cv.imdecode(jpeg, cv.IMREAD_COLOR);
    print(
      'source frame: ${probe.cols}x${probe.rows}, '
      '${jpeg.lengthInBytes} bytes JPEG, target ${inputSize}x$inputSize',
    );
    probe.dispose();

    double median(List<double> xs) {
      xs.sort();
      return xs[xs.length ~/ 2];
    }

    List<double> measure(String label, void Function() body) {
      for (var i = 0; i < warmup; i++) {
        body();
      }
      final samples = <double>[];
      for (var i = 0; i < iterations; i++) {
        final sw = Stopwatch()..start();
        body();
        sw.stop();
        samples.add(sw.elapsedMicroseconds / 1000.0);
      }
      return samples;
    }

    // OpenCV: the exact sequence lib/main.dart runs per frame.
    final openCv = measure('opencv', () {
      final src = cv.imdecode(jpeg, cv.IMREAD_COLOR);
      final scale = inputSize / (src.cols > src.rows ? src.cols : src.rows);
      final newW = (src.cols * scale).round();
      final newH = (src.rows * scale).round();
      final resized = cv.resize(src, (
        newW,
        newH,
      ), interpolation: cv.INTER_LINEAR);
      final padded = cv.copyMakeBorder(
        resized,
        0,
        inputSize - newH,
        0,
        inputSize - newW,
        cv.BORDER_CONSTANT,
        value: cv.Scalar.black,
      );
      final bytes = padded.data;
      final tensor = Float32List(inputSize * inputSize * 3);
      for (var i = 0, p = 0; i < inputSize * inputSize; i++, p += 3) {
        // BGR to RGB, normalised to [-1, 1].
        tensor[p] = bytes[p + 2] / 127.5 - 1.0;
        tensor[p + 1] = bytes[p + 1] / 127.5 - 1.0;
        tensor[p + 2] = bytes[p] / 127.5 - 1.0;
      }
      src.dispose();
      resized.dispose();
      padded.dispose();
    });

    // image: the same stages in pure Dart.
    final imagePkg = measure('image', () {
      final src = img.decodeJpg(jpeg)!;
      final scale =
          inputSize / (src.width > src.height ? src.width : src.height);
      final newW = (src.width * scale).round();
      final newH = (src.height * scale).round();
      final resized = img.copyResize(
        src,
        width: newW,
        height: newH,
        interpolation: img.Interpolation.linear,
      );
      final canvas = img.Image(width: inputSize, height: inputSize);
      img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
      img.compositeImage(canvas, resized);
      final tensor = Float32List(inputSize * inputSize * 3);
      var p = 0;
      for (var y = 0; y < inputSize; y++) {
        for (var x = 0; x < inputSize; x++) {
          final px = canvas.getPixel(x, y);
          tensor[p++] = px.r / 127.5 - 1.0;
          tensor[p++] = px.g / 127.5 - 1.0;
          tensor[p++] = px.b / 127.5 - 1.0;
        }
      }
    });

    final cvMedian = median(openCv);
    final imgMedian = median(imagePkg);
    print('');
    print('=== per-frame preprocessing, median of $iterations ===');
    print('  opencv_dart : ${cvMedian.toStringAsFixed(2)} ms');
    print('  image       : ${imgMedian.toStringAsFixed(2)} ms');
    print(
      '  ratio       : ${(imgMedian / cvMedian).toStringAsFixed(1)}x '
      'slower in pure Dart',
    );
    print('  budget at 30fps is 33.3 ms per frame, inference excluded');

    expect(cvMedian, greaterThan(0));
    expect(imgMedian, greaterThan(0));
  });
}
