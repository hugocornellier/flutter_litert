import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert_example_web/detector.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'webgpu_probe.dart';

/// End-to-end object detection through the full web engine matrix in a real
/// browser: bundled assets, dart:ui letterboxing, and YOLOv8 post-processing
/// over CompiledModel (LiteRT.js WASM, WebGPU+WASM at fp16 and fp32, and
/// strict WebGPU), LiteRtInterpreter (LiteRT.js WASM and WebGPU), and
/// Interpreter (tflite-js), plus a sweep over every bundled sample image.
///
/// GPU-flavored tests key off the [hasWebGpu] capability probe: with WebGPU
/// (a workstation browser) they assert real WebGPU compilation, without it
/// (headless CI) they assert the documented fallback or failure semantics.
///
/// The bundled `cat.jpg` sample contains two cats.
///
/// Run with:
///   chromedriver --port=4444 &
///   flutter drive --profile --driver=test_driver/integration_test.dart \
///     --target=integration_test/web_object_detection_test.dart \
///     -d web-server --browser-name=chrome
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Each test compiles a model, which includes fetching the runtime and its
  // WASM binary from a CDN on first use.
  const timeout = Timeout(Duration(minutes: 5));

  // COCO class index for "person", the subject of most bundled samples.
  const int cocoPersonClass = 0;

  Detection? bestOfClass(List<Detection> detections, int cls) {
    Detection? best;
    for (final d in detections) {
      if (d.cls == cls && (best == null || d.score > best.score)) {
        best = d;
      }
    }
    return best;
  }

  void debugPrintDetections(String label, DetectionResult result) {
    for (final d in result.detections) {
      debugPrint(
        '$label: ${cocoLabels[d.cls]} '
        '${d.score.toStringAsFixed(3)} ${d.bboxXYXY}',
      );
    }
  }

  /// The post-processor clamps boxes to the source image, so strict bounds
  /// hold for every detection; the size checks catch degenerate slivers and
  /// whole-image boxes on the subject [d].
  void expectValidGeometry(DetectionResult result, Detection d) {
    final box = d.bboxXYXY;
    final w = result.image.width.toDouble();
    final h = result.image.height.toDouble();
    for (final other in result.detections) {
      final b = other.bboxXYXY;
      expect(b[0], inInclusiveRange(0, w));
      expect(b[2], inInclusiveRange(0, w));
      expect(b[1], inInclusiveRange(0, h));
      expect(b[3], inInclusiveRange(0, h));
      expect(b[2], greaterThan(b[0]));
      expect(b[3], greaterThan(b[1]));
    }
    final area = (box[2] - box[0]) * (box[3] - box[1]);
    expect(area, greaterThan(0.02 * w * h), reason: 'subject box too small');
    expect(area, lessThan(0.95 * w * h), reason: 'subject box covers image');
  }

  void expectDetectsTheCats(DetectionResult result) {
    debugPrintDetections('detection', result);
    // Two cats plus at least part of the remotes/couch clutter.
    expect(result.detections.length, greaterThanOrEqualTo(2));
    final cat = bestOfClass(result.detections, cocoCatClass);
    expect(cat, isNotNull, reason: 'no cat detected in cat.jpg');
    expect(cat!.score, greaterThan(0.5));
    expectValidGeometry(result, cat);
  }

  Future<DetectionResult> detectWith(WebEngineConfig config) async {
    final detector = await Detector.create(config: config);
    addTearDown(detector.close);
    debugPrint('${config.label} compiled on: ${detector.backend}');
    final result = await detector.detectAsset('assets/cat.jpg');
    debugPrint('${config.label} inference: ${result.inferenceMicros}µs');
    return result;
  }

  testWidgets('CompiledModel on WASM detects the cats', (tester) async {
    final detector = await Detector.create(
      config: const WebEngineConfig(
        kind: WebEngineKind.compiledModel,
        accelerators: {Accelerator.cpu},
      ),
    );
    addTearDown(detector.close);
    expect(detector.backend, 'WASM');
    expectDetectsTheCats(await detector.detectAsset('assets/cat.jpg'));
  }, timeout: timeout);

  testWidgets('CompiledModel WebGPU request always yields a working model', (
    tester,
  ) async {
    // Without WebGPU (headless CI) this exercises the WASM fallback; with it
    // (a workstation) the model must actually compile on the GPU. Either way
    // it must detect the cats.
    final expectedBackend = await hasWebGpu() ? 'WebGPU' : 'WASM';
    final detector = await Detector.create(
      config: const WebEngineConfig(
        kind: WebEngineKind.compiledModel,
        accelerators: {Accelerator.gpu, Accelerator.cpu},
      ),
    );
    addTearDown(detector.close);
    expect(detector.backend, expectedBackend);
    expectDetectsTheCats(await detector.detectAsset('assets/cat.jpg'));
  }, timeout: timeout);

  testWidgets('CompiledModel strict WebGPU follows the capability probe', (
    tester,
  ) async {
    // {gpu} with no fallback: must genuinely compile on WebGPU where the
    // probe says it exists, and must surface the documented StateError where
    // it does not.
    if (await hasWebGpu()) {
      final detector = await Detector.create(
        config: const WebEngineConfig(
          kind: WebEngineKind.compiledModel,
          accelerators: {Accelerator.gpu},
        ),
      );
      addTearDown(detector.close);
      expect(detector.backend, 'WebGPU');
      expectDetectsTheCats(await detector.detectAsset('assets/cat.jpg'));
    } else {
      await expectLater(
        Detector.create(
          config: const WebEngineConfig(
            kind: WebEngineKind.compiledModel,
            accelerators: {Accelerator.gpu},
          ),
        ),
        throwsStateError,
      );
    }
  }, timeout: timeout);

  testWidgets('CompiledModel GPU request at fp32 detects the cats', (
    tester,
  ) async {
    // The web CompiledModel accepts precision for API parity (LiteRT.js has
    // no precision knob), so both settings must yield a working model; the
    // fp16 default is covered by the test above.
    final expectedBackend = await hasWebGpu() ? 'WebGPU' : 'WASM';
    final detector = await Detector.create(
      config: const WebEngineConfig(
        kind: WebEngineKind.compiledModel,
        accelerators: {Accelerator.gpu, Accelerator.cpu},
        precision: Precision.fp32,
      ),
    );
    addTearDown(detector.close);
    expect(detector.backend, expectedBackend);
    expectDetectsTheCats(await detector.detectAsset('assets/cat.jpg'));
  }, timeout: timeout);

  testWidgets('LiteRtInterpreter on WASM detects the cats', (tester) async {
    expectDetectsTheCats(
      await detectWith(
        const WebEngineConfig(
          kind: WebEngineKind.interpreter,
          runtime: WebInterpreterRuntime.liteRtWasm,
        ),
      ),
    );
  }, timeout: timeout);

  testWidgets(
    'LiteRtInterpreter WebGPU request always yields a working model',
    (tester) async {
      // LiteRtInterpreter.fromBytes falls back to WASM internally when the
      // WebGPU compile fails, so this must work everywhere; the resolved
      // backend must match the capability probe.
      final expectedBackend = await hasWebGpu() ? 'WebGPU' : 'WASM';
      final detector = await Detector.create(
        config: const WebEngineConfig(
          kind: WebEngineKind.interpreter,
          runtime: WebInterpreterRuntime.liteRtWebGpu,
        ),
      );
      addTearDown(detector.close);
      expect(detector.backend, expectedBackend);
      expectDetectsTheCats(await detector.detectAsset('assets/cat.jpg'));
    },
    timeout: timeout,
  );

  testWidgets('tflite-js Interpreter detects the cats', (tester) async {
    expectDetectsTheCats(
      await detectWith(
        const WebEngineConfig(
          kind: WebEngineKind.interpreter,
          runtime: WebInterpreterRuntime.tfliteJs,
        ),
      ),
    );
  }, timeout: timeout);

  testWidgets('every bundled sample detects its expected subjects', (
    tester,
  ) async {
    // One deterministic engine (CM on WASM) across all four samples. The
    // file names do not match their contents: street.jpg is a kitchen with
    // a cook, dog.jpg is a street scene with pedestrians, and people.jpg is
    // a single skier.
    final detector = await Detector.create(
      config: const WebEngineConfig(
        kind: WebEngineKind.compiledModel,
        accelerators: {Accelerator.cpu},
      ),
    );
    addTearDown(detector.close);

    const cases = <(String, int, int)>[
      ('assets/cat.jpg', cocoCatClass, 1),
      ('assets/street.jpg', cocoPersonClass, 1),
      ('assets/dog.jpg', cocoPersonClass, 2),
      ('assets/people.jpg', cocoPersonClass, 1),
    ];
    for (final (asset, cls, minCount) in cases) {
      final result = await detector.detectAsset(asset);
      debugPrintDetections(asset, result);
      expect(result.detections, isNotEmpty, reason: asset);
      final subjects = result.detections
          .where((d) => d.cls == cls && d.score > 0.5)
          .toList();
      expect(
        subjects.length,
        greaterThanOrEqualTo(minCount),
        reason:
            '$asset should contain at least $minCount ${cocoLabels[cls]}(s) '
            'above 0.5 confidence',
      );
      expectValidGeometry(result, subjects.first);
    }
  }, timeout: timeout);

  testWidgets('repeated runs on one model are stable', (tester) async {
    final detector = await Detector.create(
      config: const WebEngineConfig(
        kind: WebEngineKind.compiledModel,
        accelerators: {Accelerator.cpu},
      ),
    );
    addTearDown(detector.close);
    final first = await detector.detectAsset('assets/cat.jpg');
    final second = await detector.detectAsset('assets/cat.jpg');
    expect(second.detections.length, first.detections.length);
    final a = bestOfClass(first.detections, cocoCatClass)!;
    final b = bestOfClass(second.detections, cocoCatClass)!;
    expect(b.score, closeTo(a.score, 1e-3));
    for (var i = 0; i < 4; i++) {
      expect(b.bboxXYXY[i], closeTo(a.bboxXYXY[i], 1.0));
    }
  }, timeout: timeout);
}
