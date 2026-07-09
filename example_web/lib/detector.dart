import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_litert/flutter_litert.dart';

/// COCO class index for "cat" in YOLOv8's 80-class output.
const int cocoCatClass = 15;

/// The 80 COCO class names, index-aligned with YOLOv8 output channels 4..83.
const List<String> cocoLabels = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];

/// Which inference stack runs the model, mirroring the native example's
/// Interpreter/CompiledModel split.
enum WebEngineKind { interpreter, compiledModel }

/// Runtime behind the Interpreter engine on web: the third-party tflite-js
/// runtime or Google's LiteRT.js (WASM or WebGPU with automatic WASM
/// fallback).
enum WebInterpreterRuntime { tfliteJs, liteRtWasm, liteRtWebGpu }

/// Selected inference engine plus its options (web analogue of the native
/// example's EngineConfig).
class WebEngineConfig {
  final WebEngineKind kind;

  // Interpreter path.
  final WebInterpreterRuntime runtime;

  // CompiledModel path.
  final Set<Accelerator> accelerators;
  final Precision precision;

  const WebEngineConfig({
    this.kind = WebEngineKind.compiledModel,
    this.runtime = WebInterpreterRuntime.liteRtWasm,
    this.accelerators = const {Accelerator.gpu, Accelerator.cpu},
    this.precision = Precision.fp16,
  });

  WebEngineConfig copyWith({
    WebEngineKind? kind,
    WebInterpreterRuntime? runtime,
    Set<Accelerator>? accelerators,
    Precision? precision,
  }) => WebEngineConfig(
    kind: kind ?? this.kind,
    runtime: runtime ?? this.runtime,
    accelerators: accelerators ?? this.accelerators,
    precision: precision ?? this.precision,
  );

  String get label {
    if (kind == WebEngineKind.interpreter) {
      return 'Interpreter · ${interpreterRuntimeLabel(runtime)}';
    }
    final bool gpu = accelerators.contains(Accelerator.gpu);
    final bool cpu = accelerators.contains(Accelerator.cpu);
    final String acc = gpu ? (cpu ? 'WebGPU+WASM' : 'WebGPU') : 'WASM';
    final parts = <String>['CompiledModel', acc];
    if (gpu) parts.add(precision == Precision.fp16 ? 'fp16' : 'fp32');
    return parts.join(' · ');
  }
}

String interpreterRuntimeLabel(WebInterpreterRuntime r) => switch (r) {
  WebInterpreterRuntime.tfliteJs => 'tflite-js (WASM)',
  WebInterpreterRuntime.liteRtWasm => 'LiteRT.js (WASM)',
  WebInterpreterRuntime.liteRtWebGpu => 'LiteRT.js (WebGPU)',
};

/// One detection pass over a decoded image.
class DetectionResult {
  const DetectionResult({
    required this.image,
    required this.detections,
    required this.inferenceMicros,
  });

  /// The decoded source image (bounding boxes are in its pixel space).
  final ui.Image image;

  /// Detections above the confidence threshold, after NMS.
  final List<Detection> detections;

  /// Wall-clock time of the model run (excluding pre/post-processing).
  final int inferenceMicros;
}

/// YOLOv8n object detector that can run on any of the three web inference
/// stacks: `CompiledModel` (LiteRT.js), `LiteRtInterpreter` (LiteRT.js), or
/// `Interpreter` (tflite-js).
///
/// The whole pipeline is platform-neutral (no dart:io): assets come from the
/// bundle, preprocessing letterboxes through a dart:ui canvas, and inference
/// goes through an async runner regardless of engine, which is the only
/// shape available on the web.
class Detector {
  Detector._(this._run, this._close, this.backend);

  /// YOLOv8 square input edge.
  static const int inputSize = 640;

  // YOLOv8 80-class output geometry: [1, 84, 8400], channel-major.
  static const int _channels = 84;
  static const int _anchors = 8400;

  final Future<Float32List> Function(Float32List input) _run;
  final void Function() _close;

  /// Human-readable backend that actually compiled, e.g. 'WebGPU', 'WASM',
  /// or 'tflite-js (WASM)'. With a GPU request this reports what the runtime
  /// resolved to, so a silent WASM fallback is visible.
  final String backend;

  static Future<Uint8List> _loadBytes(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  static Future<Uint8List> _modelBytes() =>
      _loadBytes('assets/yolov8n_float32.tflite');

  /// Compiles the bundled YOLOv8n model on the engine selected by [config].
  static Future<Detector> create({
    WebEngineConfig config = const WebEngineConfig(),
  }) async {
    final bytes = await _modelBytes();
    switch (config.kind) {
      case WebEngineKind.compiledModel:
        // {gpu, cpu} tries WebGPU and falls back to WASM inside the factory.
        final model = await CompiledModel.fromBufferAsync(
          bytes,
          accelerators: config.accelerators,
          precision: config.precision,
        );
        return Detector._(
          (input) async => (await model.runAsync([input]))[0],
          model.close,
          model.accelerators.contains(Accelerator.gpu) ? 'WebGPU' : 'WASM',
        );
      case WebEngineKind.interpreter:
        switch (config.runtime) {
          case WebInterpreterRuntime.tfliteJs:
            await initializeWeb();
            final itp = await Interpreter.fromBytes(bytes);
            return Detector._(
              (input) async {
                final out = Float32List(_channels * _anchors);
                // The web Tensor.copyTo only fills ByteBuffer destinations
                // in place, so hand both sides over as raw buffers.
                itp.runForMultipleInputs([input.buffer], {0: out.buffer});
                return out;
              },
              itp.close,
              'tflite-js (WASM)',
            );
          case WebInterpreterRuntime.liteRtWasm:
          case WebInterpreterRuntime.liteRtWebGpu:
            final wantGpu =
                config.runtime == WebInterpreterRuntime.liteRtWebGpu;
            final itp = await LiteRtInterpreter.fromBytes(
              bytes,
              accelerator: wantGpu ? 'webgpu' : 'wasm',
            );
            return Detector._(
              (input) async {
                final out = Float32List(_channels * _anchors);
                await itp.runForMultipleInputs([input], {0: out});
                return out;
              },
              itp.close,
              itp.activeAccelerator == 'webgpu' ? 'WebGPU' : 'WASM',
            );
        }
    }
  }

  void close() => _close();

  /// Decodes a bundled image asset and runs detection on it.
  Future<DetectionResult> detectAsset(
    String assetPath, {
    double confThres = 0.25,
  }) async {
    final codec = await ui.instantiateImageCodec(await _loadBytes(assetPath));
    final image = (await codec.getNextFrame()).image;
    codec.dispose();
    return detectImage(image, confThres: confThres);
  }

  /// Letterboxes [image] to 640x640, runs the model, and returns detections
  /// mapped back to the image's pixel space.
  Future<DetectionResult> detectImage(
    ui.Image image, {
    double confThres = 0.25,
  }) async {
    final lb = computeLetterboxParams(
      srcWidth: image.width,
      srcHeight: image.height,
      targetWidth: inputSize,
      targetHeight: inputSize,
    );

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    const edge = inputSize * 1.0;
    canvas.drawRect(
      const ui.Rect.fromLTWH(0, 0, edge, edge),
      ui.Paint()..color = const ui.Color(0xFF727272), // YOLO's 114-gray pad.
    );
    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width * 1.0, image.height * 1.0),
      ui.Rect.fromLTWH(
        lb.padLeft * 1.0,
        lb.padTop * 1.0,
        lb.newWidth * 1.0,
        lb.newHeight * 1.0,
      ),
      ui.Paint()..filterQuality = ui.FilterQuality.low,
    );
    final letterboxed = await recorder.endRecording().toImage(
      inputSize,
      inputSize,
    );
    final rgba = (await letterboxed.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    ))!.buffer.asUint8List();
    letterboxed.dispose();

    final input = Float32List(inputSize * inputSize * 3);
    rgbaToRgbFloat32(rgba, input);

    final sw = Stopwatch()..start();
    final out = await _run(input);
    sw.stop();

    final detections = postProcessDetectionsFlat(
      out,
      channels: _channels,
      anchors: _anchors,
      channelMajor: true,
      inputWidth: inputSize,
      inputHeight: inputSize,
      r: lb.scale,
      dw: lb.padLeft,
      dh: lb.padTop,
      imageWidth: image.width,
      imageHeight: image.height,
      confThres: confThres,
      iouThres: 0.45,
      maxDet: 50,
    );

    return DetectionResult(
      image: image,
      detections: detections,
      inferenceMicros: sw.elapsedMicroseconds,
    );
  }
}
