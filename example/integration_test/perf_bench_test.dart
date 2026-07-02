// On-device performance benchmark for A/B comparison across branch
// revisions. Self-contained: the 544-byte add model is embedded, utility
// fixtures are synthetic, and every bench fails soft so one broken area
// does not hide the others. Results land in integration_test reportData
// and are printed as a single PERFBENCH_JSON line.
//
// Run with:
//   flutter drive --driver=test_driver/integration_test.dart \
//     --target=integration_test/perf_bench_test.dart --profile -d <device>
//
// ignore_for_file: avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/src/compiled_model/compiled_model_native.dart';
import 'package:flutter_litert/src/native/interpreter.dart';
import 'package:flutter_litert/src/util/image_tensor_utils.dart';
import 'package:flutter_litert/src/util/model_output_utils.dart';
import 'package:flutter_litert/src/util/yuv_conversion.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const _addModelB64 =
    'JAAAAFRGTDMAAAAAAAAAABQAGAAEAAgADAAAABAAAAAAABQAFAAAAAMAAADkAQAAmAAAAIAAAAAEAAAAAQAAABAAAAAAAAoAEAAEAAgADAAKAAAAPAAAABwAAAAEAAAADwAAAHNlcnZpbmdfZGVmYXVsdAABAAAABAAAAOT///8IAAAAAgAAAAEAAAB4AAAAAQAAAAwAAAAIAAwABAAIAAgAAAAIAAAAAQAAAAEAAABhAAAAAQAAAAQAAACk/v//AAAAAAAAAAABAAAAEAAAAAwAFAAEAAgADAAQAAwAAACUAAAAiAAAAHwAAAAEAAAAAgAAAEQAAAAEAAAA0v///wAAAAsYAAAADAAAAAQAAAD4/v//AQAAAAIAAAACAAAAAAAAAAEAAAAAAA4AFAAAAAgADAAHABAADgAAAAAAAAsYAAAADAAAAAQAAAA0////AQAAAAAAAAACAAAAAQAAAAEAAAABAAAAAgAAAAEAAAABAAAAAwAAAHAAAAA0AAAABAAAAKj///8UAAAABAAAAAYAAABvdXRwdXQAAAQAAAABAAAACAAAAAgAAAADAAAA1P///xQAAAAEAAAABQAAAGlucHV0AAAABAAAAAEAAAAIAAAACAAAAAMAAAAMAAwABAAAAAAACAAMAAAAEAAAAAQAAAADAAAAYWRkAAQAAAABAAAACAAAAAgAAAADAAAAAQAAAAgAAAAEAAQABAAAAA==';

double _sink = 0;

final Stopwatch _clock = Stopwatch();
final double _nsPerTick = 1e9 / _clock.frequency;

const _samples = 40;
const _targetBatchMs = 4.0;

Map<String, double> _bench(void Function() body) {
  body();
  _clock
    ..reset()
    ..start();
  var warmupIters = 0;
  while (_clock.elapsedMilliseconds < 300) {
    body();
    warmupIters++;
  }
  _clock.stop();
  final warmupNsPerOp =
      _clock.elapsedTicks * _nsPerTick / math.max(1, warmupIters);
  final batch = math.max(
    1,
    (_targetBatchMs * 1e6 / math.max(1, warmupNsPerOp)).round(),
  );
  final samples = <double>[];
  for (var s = 0; s < _samples; s++) {
    _clock
      ..reset()
      ..start();
    for (var i = 0; i < batch; i++) {
      body();
    }
    _clock.stop();
    samples.add(_clock.elapsedTicks * _nsPerTick / batch);
  }
  samples.sort();
  return {
    'median_ns': samples[samples.length ~/ 2],
    'p90_ns': samples[(samples.length * 9) ~/ 10],
    'batch': batch.toDouble(),
  };
}

Future<Map<String, double>> _benchAsync(Future<void> Function() body) async {
  await body();
  _clock
    ..reset()
    ..start();
  var warmupIters = 0;
  while (_clock.elapsedMilliseconds < 300) {
    await body();
    warmupIters++;
  }
  _clock.stop();
  final warmupNsPerOp =
      _clock.elapsedTicks * _nsPerTick / math.max(1, warmupIters);
  final batch = math.max(
    1,
    (_targetBatchMs * 1e6 / math.max(1, warmupNsPerOp)).round(),
  );

  final tickerClock = Stopwatch()..start();
  var lastTickNs = tickerClock.elapsedTicks * _nsPerTick;
  var maxGapNs = 0.0;
  final ticker = Timer.periodic(const Duration(milliseconds: 1), (_) {
    final now = tickerClock.elapsedTicks * _nsPerTick;
    final gap = now - lastTickNs;
    lastTickNs = now;
    if (gap > maxGapNs) maxGapNs = gap;
  });

  final samples = <double>[];
  final stalls = <double>[];
  for (var s = 0; s < _samples; s++) {
    maxGapNs = 0;
    lastTickNs = tickerClock.elapsedTicks * _nsPerTick;
    _clock
      ..reset()
      ..start();
    for (var i = 0; i < batch; i++) {
      await body();
    }
    _clock.stop();
    samples.add(_clock.elapsedTicks * _nsPerTick / batch);
    final endGapNs = tickerClock.elapsedTicks * _nsPerTick - lastTickNs;
    stalls.add(math.max(0, math.max(maxGapNs, endGapNs) - 1e6));
  }
  ticker.cancel();
  samples.sort();
  stalls.sort();
  return {
    'median_ns': samples[samples.length ~/ 2],
    'stall_median_ns': stalls[stalls.length ~/ 2],
    'batch': batch.toDouble(),
  };
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('on-device perf bench', (tester) async {
    final results = <String, Object>{};

    await tester.runAsync(() async {
      final addModel = base64Decode(_addModelB64);

      void tryBench(String name, void Function() Function() setup) {
        try {
          results[name] = _bench(setup());
        } catch (e) {
          results[name] = {'error': e.toString()};
        }
      }

      // Interpreter Float32List path.
      Interpreter? interp;
      try {
        interp = Interpreter.fromBuffer(addModel)..allocateTensors();
        final len = interp.getInputTensor(0).numBytes() ~/ 4;
        final input = Float32List(len);
        for (var i = 0; i < len; i++) {
          input[i] = i * 0.001;
        }
        final out = Float32List(len);
        final it = interp;
        tryBench('interp_add_f32', () {
          return () {
            it.run(input, out);
            _sink += out[0];
          };
        });
      } catch (e) {
        results['interp_add_f32'] = {'error': e.toString()};
      }

      // CompiledModel managed and hostMemory paths.
      for (final (name, mode) in [
        ('cm_add_managed_run', TensorBufferMode.managed),
        ('cm_add_host_run', TensorBufferMode.hostMemory),
      ]) {
        try {
          final cm = CompiledModel.fromBuffer(addModel, tensorBufferMode: mode);
          final len = cm.inputByteSizes[0] ~/ 4;
          final input = Float32List(len);
          for (var i = 0; i < len; i++) {
            input[i] = i * 0.001;
          }
          tryBench(name, () {
            return () {
              final out = cm.run([input]);
              _sink += out[0][0];
            };
          });
          if (mode == TensorBufferMode.hostMemory) {
            try {
              results['cm_add_async_run'] = await _benchAsync(() async {
                final out = await cm.runAsync([input]);
                _sink += out[0][0];
              });
            } catch (e) {
              results['cm_add_async_run'] = {'error': e.toString()};
            }
          }
          cm.close();
        } catch (e) {
          results[name] = {'error': e.toString()};
        }
      }
      interp?.close();

      // Utility: YOLOv8-shaped decode, channel-major.
      try {
        const yoloCh = 84, yoloAnchors = 8400;
        final yoloOut = Float32List(yoloCh * yoloAnchors);
        final rng = math.Random(11);
        for (var a = 0; a < yoloAnchors; a++) {
          yoloOut[a] = 100 + rng.nextDouble() * 440;
          yoloOut[yoloAnchors + a] = 100 + rng.nextDouble() * 440;
          yoloOut[2 * yoloAnchors + a] = 20 + rng.nextDouble() * 180;
          yoloOut[3 * yoloAnchors + a] = 20 + rng.nextDouble() * 180;
          for (var k = 4; k < yoloCh; k++) {
            yoloOut[k * yoloAnchors + a] = -9 + rng.nextDouble();
          }
        }
        for (var n = 0; n < 60; n++) {
          final a = rng.nextInt(yoloAnchors);
          final k = 4 + rng.nextInt(yoloCh - 4);
          yoloOut[k * yoloAnchors + a] = 2 + rng.nextDouble() * 4;
        }
        tryBench('util_postprocess_yolo', () {
          return () {
            final dets = postProcessDetectionsFlat(
              yoloOut,
              channels: yoloCh,
              anchors: yoloAnchors,
              channelMajor: true,
              inputWidth: 640,
              inputHeight: 640,
              r: 0.5,
              dw: 0,
              dh: 140,
              imageWidth: 1280,
              imageHeight: 720,
              confThres: 0.25,
              iouThres: 0.45,
              maxDet: 100,
            );
            _sink += dets.length.toDouble();
          };
        });
      } catch (e) {
        results['util_postprocess_yolo'] = {'error': e.toString()};
      }

      // Utility: normalization.
      try {
        const pixels = 256 * 256;
        final bytes = Uint8List(pixels * 3);
        final rng = math.Random(7);
        for (var i = 0; i < bytes.length; i++) {
          bytes[i] = rng.nextInt(256);
        }
        final buf = Float32List(pixels * 3);
        tryBench('util_bgr_signed_f32', () {
          return () {
            final t = bgrBytesToSignedFloat32(
              bytes: bytes,
              totalPixels: pixels,
              buffer: buf,
            );
            _sink += t[0];
          };
        });
      } catch (e) {
        results['util_bgr_signed_f32'] = {'error': e.toString()};
      }

      // Utility: 720p YUV pack (allocating form; exists on all revisions).
      try {
        const w = 1280, h = 720;
        const yStride = w + 64, uvStride = w ~/ 2 + 32;
        final y = Uint8List(yStride * h);
        final u = Uint8List(uvStride * (h ~/ 2));
        final v = Uint8List(uvStride * (h ~/ 2));
        tryBench('util_yuv_pack_720p', () {
          return () {
            final packed = packYuv420(
              width: w,
              height: h,
              y: (bytes: y, rowStride: yStride, pixelStride: 1),
              u: (bytes: u, rowStride: uvStride, pixelStride: 1),
              v: (bytes: v, rowStride: uvStride, pixelStride: 1),
            );
            _sink += packed!.bytes[0].toDouble();
          };
        });
      } catch (e) {
        results['util_yuv_pack_720p'] = {'error': e.toString()};
      }
    });

    results['sink'] = _sink;
    binding.reportData = <String, dynamic>{'perfbench': results};
    print('PERFBENCH_JSON: ${jsonEncode(results)}');
    expect(results, isNotEmpty);
  });
}
