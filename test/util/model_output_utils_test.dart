import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/src/util/model_output_utils.dart';
import 'package:flutter_litert/src/util/math_utils.dart';
import 'package:flutter_test/flutter_test.dart';

/// Builds an anchor-major flat output buffer from per-anchor rows.
Float32List _flat(List<List<double>> rows) {
  final out = Float32List(rows.length * rows[0].length);
  var i = 0;
  for (final row in rows) {
    for (final v in row) {
      out[i++] = v;
    }
  }
  return out;
}

void main() {
  group('postProcessDetectionsFlat threshold pruning', () {
    // No-objectness layout (channels == 84): score is sigmoid of the best
    // class logit. Anchors sit below, exactly at, and above confThres; the
    // logit-space prune must keep the boundary anchor (score == confThres
    // passes the `score < confThres` check).
    test('keeps the exact-threshold anchor and drops sub-threshold', () {
      const channels = 84;
      final rows = <List<double>>[];
      // Distinct, non-overlapping locations so NMS keeps every survivor and
      // the assertion isolates the confidence threshold behavior.
      var cx = 100.0;
      for (final classLogit in [-2.0, 0.0, 2.0]) {
        rows.add([
          cx, 320.0, 50.0, 50.0, // cx, cy, w, h (pixel scale)
          classLogit,
          ...List.filled(channels - 5, -20.0),
        ]);
        cx += 200.0;
      }
      final dets = postProcessDetectionsFlat(
        _flat(rows),
        channels: channels,
        anchors: rows.length,
        channelMajor: false,
        inputWidth: 640,
        inputHeight: 640,
        r: 1.0,
        dw: 0,
        dh: 0,
        imageWidth: 640,
        imageHeight: 640,
        confThres: 0.5, // == sigmoid(0.0), the boundary anchor's score
        iouThres: 0.99, // boxes overlap fully; keep both survivors
        maxDet: 10,
      );
      expect(dets, hasLength(2));
      expect(dets.map((d) => d.score).toList(), [
        closeTo(sigmoid(2.0), 1e-6),
        closeTo(0.5, 1e-6),
      ]);
      expect(dets.every((d) => d.cls == 0), isTrue);
    });

    // Objectness layout (channels != 84): score is
    // sigmoid(classLogit) * sigmoid(objLogit). An anchor whose class logit
    // clears the threshold alone but whose product falls below it must
    // still be dropped, which proves the prune only removes anchors the
    // full check would also remove.
    test('objectness product still applies after the prune', () {
      const channels = 6; // 4 box + objectness + 1 class
      final rows = <List<double>>[
        // sigmoid(2.0) ~ 0.881 passes alone; * sigmoid(-2.0) ~ 0.119 fails.
        [100.0, 100.0, 50.0, 50.0, -2.0, 2.0],
        // sigmoid(2.0) * sigmoid(4.0) ~ 0.865 passes.
        [400.0, 400.0, 50.0, 50.0, 4.0, 2.0],
      ];
      final dets = postProcessDetectionsFlat(
        _flat(rows),
        channels: channels,
        anchors: rows.length,
        channelMajor: false,
        inputWidth: 640,
        inputHeight: 640,
        r: 1.0,
        dw: 0,
        dh: 0,
        imageWidth: 640,
        imageHeight: 640,
        confThres: 0.5,
        iouThres: 0.5,
        maxDet: 10,
      );
      expect(dets, hasLength(1));
      expect(dets.single.score, closeTo(sigmoid(2.0) * sigmoid(4.0), 1e-6));
      expect(dets.single.bboxXYXY[0], closeTo(375.0, 1e-6));
    });
  });

  test('postProcessDetectionsFlat rejects a too-short output buffer', () {
    expect(
      () => postProcessDetectionsFlat(
        Float32List(10),
        channels: 84,
        anchors: 8400,
        channelMajor: true,
        inputWidth: 640,
        inputHeight: 640,
        r: 1.0,
        dw: 0,
        dh: 0,
        imageWidth: 640,
        imageHeight: 640,
        confThres: 0.5,
        iouThres: 0.5,
        maxDet: 10,
      ),
      throwsArgumentError,
    );
  });

  group('postProcessDetectionsFlat layout equivalence', () {
    // The channel-major decode uses a SIMD argmax when anchors is a
    // multiple of 4 and a scalar strided loop otherwise; the anchor-major
    // decode is always scalar. All three must produce identical detections
    // for the same logical tensor.
    List<Detection> run(
      Float32List out,
      int channels,
      int anchors, {
      required bool channelMajor,
    }) {
      return postProcessDetectionsFlat(
        out,
        channels: channels,
        anchors: anchors,
        channelMajor: channelMajor,
        inputWidth: 640,
        inputHeight: 640,
        r: 0.5,
        dw: 0,
        dh: 140,
        imageWidth: 1280,
        imageHeight: 720,
        confThres: 0.3,
        iouThres: 0.45,
        maxDet: 50,
      );
    }

    void expectSameDetections(List<Detection> a, List<Detection> b) {
      expect(b, hasLength(a.length));
      for (var i = 0; i < a.length; i++) {
        expect(b[i].cls, a[i].cls);
        expect(b[i].score, closeTo(a[i].score, 1e-9));
        for (var c = 0; c < 4; c++) {
          expect(b[i].bboxXYXY[c], closeTo(a[i].bboxXYXY[c], 1e-9));
        }
      }
    }

    (Float32List, Float32List) buildBothLayouts(
      int channels,
      int anchors,
      math.Random rng,
    ) {
      final channelMajorOut = Float32List(channels * anchors);
      final anchorMajorOut = Float32List(channels * anchors);
      for (var a = 0; a < anchors; a++) {
        for (var c = 0; c < channels; c++) {
          final double v;
          if (c == 0 || c == 1) {
            v = 50 + rng.nextDouble() * 540;
          } else if (c == 2 || c == 3) {
            v = 20 + rng.nextDouble() * 200;
          } else {
            // Mostly sub-threshold logits with occasional strong hits, and
            // deliberate exact ties to pin down first-wins argmax.
            final roll = rng.nextInt(100);
            v = roll < 3
                ? 1.5
                : roll < 6
                ? -1 + rng.nextDouble() * 4
                : -9 + rng.nextDouble();
          }
          channelMajorOut[c * anchors + a] = v;
          anchorMajorOut[a * channels + c] = v;
        }
      }
      return (channelMajorOut, anchorMajorOut);
    }

    test('SIMD channel-major matches scalar anchor-major (84x512)', () {
      final (cm, am) = buildBothLayouts(84, 512, math.Random(3));
      expectSameDetections(
        run(am, 84, 512, channelMajor: false),
        run(cm, 84, 512, channelMajor: true),
      );
    });

    test('scalar fallback channel-major matches anchor-major (84x511)', () {
      final (cm, am) = buildBothLayouts(84, 511, math.Random(5));
      expectSameDetections(
        run(am, 84, 511, channelMajor: false),
        run(cm, 84, 511, channelMajor: true),
      );
    });

    test('objectness layout equivalence (6x512)', () {
      final (cm, am) = buildBothLayouts(6, 512, math.Random(9));
      expectSameDetections(
        run(am, 6, 512, channelMajor: false),
        run(cm, 6, 512, channelMajor: true),
      );
    });
  });
}
