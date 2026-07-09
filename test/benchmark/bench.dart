// ignore_for_file: avoid_print
//
// Batched nanosecond-resolution micro-benchmark harness for the native
// hot paths (CompiledModel and Interpreter).
//
// Runs on the plain Dart VM so it can be compiled AOT:
//   JIT:  dart run test/benchmark/bench.dart
//   AOT:  dart compile exe test/benchmark/bench.dart -o bench && ./bench
//
// Always run from the repository root: native libraries and model assets
// are resolved relative to the working directory. Results append to a
// JSONL file with --out for A/B comparison via test/benchmark/compare.dart.
//
// Timing model: the wall clock ticks in microseconds, so a sample is one
// timed batch of N iterations and ns/op = batch_time / N. N is calibrated
// per bench so a batch lasts a few milliseconds, which makes clock
// quantization negligible.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/src/compiled_model/compiled_model_native.dart';
import 'package:flutter_litert/src/native/interpreter.dart';
import 'package:flutter_litert/src/util/image_tensor_utils.dart';
import 'package:flutter_litert/src/util/model_output_utils.dart';
import 'package:flutter_litert/src/util/nms_utils.dart';
import 'package:flutter_litert/src/util/yuv_conversion.dart';

const _addModel = 'test/assets/add.tflite';
const _faceModel = 'test/assets/face_detection_short_range.tflite';

// Accumulates a value derived from every benchmark body so AOT cannot
// dead-code-eliminate the work. Printed at the end.
double _sink = 0;

class _Options {
  _Options({
    required this.filter,
    required this.samples,
    required this.targetBatchMs,
    required this.out,
    required this.label,
  });

  final String? filter;
  final int samples;
  final double targetBatchMs;
  final String? out;
  final String label;
}

_Options _parseArgs(List<String> args) {
  String? filter;
  var samples = 60;
  var targetBatchMs = 4.0;
  String? out;
  var label = '';
  for (final arg in args) {
    if (arg.startsWith('--filter=')) {
      filter = arg.substring('--filter='.length);
    } else if (arg.startsWith('--samples=')) {
      samples = int.parse(arg.substring('--samples='.length));
    } else if (arg.startsWith('--target-ms=')) {
      targetBatchMs = double.parse(arg.substring('--target-ms='.length));
    } else if (arg.startsWith('--out=')) {
      out = arg.substring('--out='.length);
    } else if (arg.startsWith('--label=')) {
      label = arg.substring('--label='.length);
    } else {
      stderr.writeln('Unknown argument: $arg');
      exit(64);
    }
  }
  return _Options(
    filter: filter,
    samples: samples,
    targetBatchMs: targetBatchMs,
    out: out,
    label: label,
  );
}

class _BenchResult {
  _BenchResult(this.name, this.batch, this.samplesNsPerOp);

  final String name;
  final int batch;
  final List<double> samplesNsPerOp;

  double percentile(double p) {
    final s = [...samplesNsPerOp]..sort();
    final idx = ((s.length - 1) * p / 100).round();
    return s[idx];
  }
}

final Stopwatch _clock = Stopwatch();
final double _nsPerTick = 1e9 / _clock.frequency;

_BenchResult _bench(String name, void Function() body, _Options opts) {
  body(); // Fail fast before timing.

  // Warmup for ~300ms so JIT and caches reach steady state.
  _clock
    ..reset()
    ..start();
  var warmupIters = 0;
  while (_clock.elapsedMilliseconds < 300) {
    body();
    warmupIters++;
  }
  _clock.stop();

  // Calibrate batch size N so one timed batch lasts ~targetBatchMs.
  final warmupNsPerOp =
      _clock.elapsedTicks * _nsPerTick / math.max(1, warmupIters);
  final batch = math.max(
    1,
    (opts.targetBatchMs * 1e6 / math.max(1, warmupNsPerOp)).round(),
  );

  final samples = <double>[];
  for (var s = 0; s < opts.samples; s++) {
    _clock
      ..reset()
      ..start();
    for (var i = 0; i < batch; i++) {
      body();
    }
    _clock.stop();
    samples.add(_clock.elapsedTicks * _nsPerTick / batch);
  }
  return _BenchResult(name, batch, samples);
}

/// Times an awaited async body and, concurrently, measures how responsive
/// the main isolate's event loop stays while the body runs.
///
/// A 1 ms periodic timer records the gap between consecutive fires; a body
/// that blocks the isolate in native code delays the timer, so
/// max(gap) - 1 ms approximates the worst single stall in that sample
/// window. Returns two result rows: `<name>` (ns/op of the awaited call)
/// and `<name>_stall` (worst event-loop stall in ns per sample window).
Future<List<_BenchResult>> _benchAsync(
  String name,
  Future<void> Function() body,
  _Options opts,
) async {
  await body(); // Fail fast before timing.

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
    (opts.targetBatchMs * 1e6 / math.max(1, warmupNsPerOp)).round(),
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
  for (var s = 0; s < opts.samples; s++) {
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
    // A starved timer never fires inside the window, leaving maxGapNs at 0;
    // the gap still pending at the end of the window is the floor of the
    // real stall, so take whichever is worse.
    final endGapNs = tickerClock.elapsedTicks * _nsPerTick - lastTickNs;
    stalls.add(math.max(0, math.max(maxGapNs, endGapNs) - 1e6));
  }
  ticker.cancel();
  return [
    _BenchResult(name, batch, samples),
    _BenchResult('${name}_stall', batch, stalls),
  ];
}

String _gitShort(List<String> args) {
  try {
    final r = Process.runSync('git', args);
    if (r.exitCode != 0) return '';
    return (r.stdout as String).trim();
  } catch (_) {
    return '';
  }
}

Future<void> main(List<String> args) async {
  final opts = _parseArgs(args);
  final mode = Platform.script.path.endsWith('.dart') ? 'jit' : 'aot';
  final sha = _gitShort(['rev-parse', '--short', 'HEAD']);
  final dirty = _gitShort(['status', '--porcelain']).isNotEmpty;

  final benches = <String, void Function()>{};

  // ---- CompiledModel: tiny add model, managed buffers (default mode) ----
  final cmManaged = CompiledModel.fromFile(_addModel);
  final cmInLen = cmManaged.inputByteSizes[0] ~/ 4;
  final cmInput = Float32List(cmInLen);
  for (var i = 0; i < cmInput.length; i++) {
    cmInput[i] = i * 0.001;
  }
  benches['cm_add_managed_run'] = () {
    final out = cmManaged.run([cmInput]);
    _sink += out[0][0];
  };

  // ---- CompiledModel: tiny add model, hostMemory buffers ----
  final cmHost = CompiledModel.fromFile(
    _addModel,
    tensorBufferMode: TensorBufferMode.hostMemory,
  );
  benches['cm_add_host_run'] = () {
    final out = cmHost.run([cmInput]);
    _sink += out[0][0];
  };
  benches['cm_add_host_zero'] = () {
    cmHost.writeInput(0, (dst) => dst.setAll(0, cmInput));
    cmHost.dispatch();
    _sink += cmHost.readOutput(0, (o) => o[0]);
  };

  // ---- CompiledModel: real per-frame model (sanity, low sensitivity) ----
  final cmFace = CompiledModel.fromFile(
    _faceModel,
    tensorBufferMode: TensorBufferMode.hostMemory,
  );
  final faceInLen = cmFace.inputByteSizes[0] ~/ 4;
  final faceInput = Float32List(faceInLen);
  final rng = math.Random(42);
  for (var i = 0; i < faceInput.length; i++) {
    faceInput[i] = rng.nextDouble() * 2 - 1;
  }
  benches['cm_face_host_run'] = () {
    final out = cmFace.run([faceInput]);
    _sink += out[0][0] + out[1][0];
  };

  // ---- Interpreter: tiny add model, Float32List in/out fast path ----
  final interpAdd = Interpreter.fromFile(File(_addModel))..allocateTensors();
  final interpAddOut = Float32List(cmInLen);
  benches['interp_add_f32'] = () {
    interpAdd.run(cmInput, interpAddOut);
    _sink += interpAddOut[0];
  };

  // ---- Interpreter: real model, multi-output fast path ----
  final interpFace = Interpreter.fromFile(File(_faceModel))..allocateTensors();
  final faceOut0 = Float32List(896 * 16);
  final faceOut1 = Float32List(896 * 1);
  final faceOutputs = <int, Object>{0: faceOut0, 1: faceOut1};
  benches['interp_face_f32'] = () {
    interpFace.runForMultipleInputs([faceInput], faceOutputs);
    _sink += faceOut0[0] + faceOut1[0];
  };

  // ---- Shared per-frame utilities used by the detection packages ----
  // Normalization: 256x256 model input, reused output buffers (the pattern
  // the detection isolates use).
  const normPixels = 256 * 256;
  final bgrBytes = Uint8List(normPixels * 3);
  final rgbaBytes = Uint8List(normPixels * 4);
  final normRng = math.Random(7);
  for (var i = 0; i < bgrBytes.length; i++) {
    bgrBytes[i] = normRng.nextInt(256);
  }
  for (var i = 0; i < rgbaBytes.length; i++) {
    rgbaBytes[i] = normRng.nextInt(256);
  }
  final bgrOut = Float32List(normPixels * 3);
  final rgbaOut = Float32List(normPixels * 3);
  benches['util_bgr_signed_f32'] = () {
    final t = bgrBytesToSignedFloat32(
      bytes: bgrBytes,
      totalPixels: normPixels,
      buffer: bgrOut,
    );
    _sink += t[0];
  };
  benches['util_rgba_f32'] = () {
    rgbaToRgbFloat32(rgbaBytes, rgbaOut);
    _sink += rgbaOut[0];
  };

  // YUV pack: 720p planar I420 with padded row strides, so the row-copy path
  // (not the single-memcpy shortcut) is exercised.
  const yuvW = 1280, yuvH = 720;
  const yStride = yuvW + 64, uvStride = yuvW ~/ 2 + 32;
  final yPlane = Uint8List(yStride * yuvH);
  final uPlane = Uint8List(uvStride * (yuvH ~/ 2));
  final vPlane = Uint8List(uvStride * (yuvH ~/ 2));
  benches['util_yuv_pack_720p'] = () {
    final packed = packYuv420(
      width: yuvW,
      height: yuvH,
      y: (bytes: yPlane, rowStride: yStride, pixelStride: 1),
      u: (bytes: uPlane, rowStride: uvStride, pixelStride: 1),
      v: (bytes: vPlane, rowStride: uvStride, pixelStride: 1),
    );
    _sink += packed!.bytes[0].toDouble();
  };
  final yuvReuse = Uint8List(yuvW * yuvH * 3 ~/ 2);
  benches['util_yuv_pack_720p_reuse'] = () {
    final packed = packYuv420(
      width: yuvW,
      height: yuvH,
      y: (bytes: yPlane, rowStride: yStride, pixelStride: 1),
      u: (bytes: uPlane, rowStride: uvStride, pixelStride: 1),
      v: (bytes: vPlane, rowStride: uvStride, pixelStride: 1),
      into: yuvReuse,
    );
    _sink += packed!.bytes[0].toDouble();
  };

  // YOLOv8-shaped detection decode: 84 channels x 8400 anchors,
  // channel-major, 640x640 letterbox onto a 1280x720 image. Logits sit well
  // below threshold except ~60 boosted anchors, which is a realistic
  // post-threshold survivor count.
  const yoloCh = 84, yoloAnchors = 8400;
  final yoloOut = Float32List(yoloCh * yoloAnchors);
  final yoloRng = math.Random(11);
  for (var a = 0; a < yoloAnchors; a++) {
    yoloOut[a] = 100 + yoloRng.nextDouble() * 440; // cx
    yoloOut[yoloAnchors + a] = 100 + yoloRng.nextDouble() * 440; // cy
    yoloOut[2 * yoloAnchors + a] = 20 + yoloRng.nextDouble() * 180; // w
    yoloOut[3 * yoloAnchors + a] = 20 + yoloRng.nextDouble() * 180; // h
    for (var k = 4; k < yoloCh; k++) {
      yoloOut[k * yoloAnchors + a] = -9 + yoloRng.nextDouble();
    }
  }
  for (var n = 0; n < 60; n++) {
    final a = yoloRng.nextInt(yoloAnchors);
    final k = 4 + yoloRng.nextInt(yoloCh - 4);
    yoloOut[k * yoloAnchors + a] = 2 + yoloRng.nextDouble() * 4;
  }
  benches['util_postprocess_yolo'] = () {
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

  // NMS alone: 120 clustered candidate boxes.
  final nmsBoxes = <List<double>>[];
  final nmsScores = <double>[];
  final nmsRng = math.Random(13);
  for (var i = 0; i < 120; i++) {
    final cx = 100 + nmsRng.nextInt(6) * 180 + nmsRng.nextDouble() * 30;
    final cy = 100 + nmsRng.nextInt(3) * 180 + nmsRng.nextDouble() * 30;
    final w = 80 + nmsRng.nextDouble() * 60;
    final h = 80 + nmsRng.nextDouble() * 60;
    nmsBoxes.add(<double>[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    nmsScores.add(0.3 + nmsRng.nextDouble() * 0.7);
  }
  benches['util_nms_120'] = () {
    final keep = nms(nmsBoxes, nmsScores, iouThres: 0.45, maxDet: 100);
    _sink += keep.length.toDouble();
  };

  // ---- Async dispatch: awaited latency and main-isolate responsiveness ----
  final asyncBenches = <String, Future<void> Function()>{};
  asyncBenches['cm_add_async_run'] = () async {
    final out = await cmHost.runAsync([cmInput]);
    _sink += out[0][0];
  };
  asyncBenches['cm_face_async_run'] = () async {
    final out = await cmFace.runAsync([faceInput]);
    _sink += out[0][0] + out[1][0];
  };

  final results = <_BenchResult>[];
  void report(_BenchResult r) {
    results.add(r);
    print(
      '${r.name.padRight(24)} median ${r.percentile(50).toStringAsFixed(1).padLeft(12)} ns/op   '
      'p10 ${r.percentile(10).toStringAsFixed(1).padLeft(12)}   '
      'p90 ${r.percentile(90).toStringAsFixed(1).padLeft(12)}   '
      '(batch ${r.batch}, ${r.samplesNsPerOp.length} samples)',
    );
  }

  for (final entry in benches.entries) {
    if (opts.filter != null && !entry.key.contains(opts.filter!)) continue;
    report(_bench(entry.key, entry.value, opts));
  }
  for (final entry in asyncBenches.entries) {
    if (opts.filter != null && !entry.key.contains(opts.filter!)) continue;
    (await _benchAsync(entry.key, entry.value, opts)).forEach(report);
  }

  cmManaged.close();
  cmHost.close();
  cmFace.close();
  interpAdd.close();
  interpFace.close();

  if (opts.out != null) {
    final f = File(opts.out!);
    final sinkLines = results
        .map(
          (r) => jsonEncode({
            'bench': r.name,
            'mode': mode,
            'sha': sha,
            'dirty': dirty,
            'label': opts.label,
            'batch': r.batch,
            'samples': r.samplesNsPerOp,
          }),
        )
        .join('\n');
    f.writeAsStringSync('$sinkLines\n', mode: FileMode.append);
  }

  print('mode=$mode sha=$sha dirty=$dirty sink=$_sink');
}
