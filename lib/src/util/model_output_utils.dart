import 'dart:math' as math;
import 'dart:typed_data';

import 'math_utils.dart';
import 'nms_utils.dart';
import 'detection_utils.dart';

/// A single object detection result from a detection model.
///
/// Contains the detected class ID, confidence score, and bounding box coordinates.
class Detection {
  /// Detected class ID.
  final int cls;

  /// Confidence score for the detection (0.0 to 1.0).
  final double score;

  /// Bounding box in XYXY format `[x1, y1, x2, y2]` in pixel coordinates.
  final List<double> bboxXYXY;

  Detection({required this.cls, required this.score, required this.bboxXYXY});
}

/// Decodes raw detection model outputs and splits each row into xywh, rest, and C.
///
/// Returns a list of maps with keys:
/// - `xywh`: first 4 values (bounding box center-xy + width/height)
/// - `rest`: remaining values (objectness + class logits)
/// - `C`: total number of channels in the row
List<Map<String, dynamic>> decodeAndSplitOutputs(List<dynamic> outputs) {
  final List<List<double>> out = decodeDetectionOutputs(outputs);
  final int channels = out[0].length;
  return out
      .map(
        (row) => {
          'xywh': row.sublist(0, 4),
          'rest': row.sublist(4),
          'C': channels,
        },
      )
      .toList();
}

/// Post-processes detection model outputs into [Detection] results.
///
/// Decodes model outputs, applies confidence filtering, optional class filtering,
/// top-k pre-NMS selection, NMS, and coordinate transformation from letterbox
/// space to original image coordinates.
///
/// Parameters:
/// - [outputs]: Raw model output tensors.
/// - [inputWidth]: Width of the model input tensor (used for coordinate de-normalization).
/// - [inputHeight]: Height of the model input tensor (used for coordinate de-normalization).
/// - [r]: Letterbox scale ratio.
/// - [dw]: Horizontal letterbox padding in pixels.
/// - [dh]: Vertical letterbox padding in pixels.
/// - [imageWidth]: Original image width for coordinate clamping.
/// - [imageHeight]: Original image height for coordinate clamping.
/// - [confThres]: Minimum confidence score to keep a detection.
/// - [iouThres]: IoU threshold for NMS.
/// - [topkPreNms]: Number of top candidates to keep before NMS (0 = auto-scale).
/// - [maxDet]: Maximum number of detections to return after NMS.
/// - [filterClassId]: If non-null, only detections with this class ID are kept.
/// - [scoresAreProbabilities]: Whether class/objectness values are already
///   probabilities. Defaults to false, preserving the logits + sigmoid
///   contract used by existing callers.
List<Detection> postProcessDetections({
  required List<dynamic> outputs,
  required int inputWidth,
  required int inputHeight,
  required double r,
  required int dw,
  required int dh,
  required int imageWidth,
  required int imageHeight,
  required double confThres,
  required double iouThres,
  required int topkPreNms,
  required int maxDet,
  int? filterClassId,
  bool scoresAreProbabilities = false,
}) {
  final List<Map<String, dynamic>> decoded = decodeAndSplitOutputs(outputs);
  final List<int> clsIds = <int>[];
  final List<double> scores = <double>[];
  final List<List<double>> xywhs = <List<double>>[];

  for (final Map<String, dynamic> row in decoded) {
    final int C = row['C'] as int;
    final List<double> xywh = (row['xywh'] as List)
        .map((v) => (v as num).toDouble())
        .toList();
    final List<double> rest = (row['rest'] as List)
        .map((v) => (v as num).toDouble())
        .toList();

    if (C == 84) {
      int argMax = 0;
      double best = -1e9;
      for (int i = 0; i < rest.length; i++) {
        final double s = scoresAreProbabilities ? rest[i] : sigmoid(rest[i]);
        if (s > best) {
          best = s;
          argMax = i;
        }
      }
      scores.add(best);
      clsIds.add(argMax);
      xywhs.add(xywh);
    } else {
      final double obj = scoresAreProbabilities ? rest[0] : sigmoid(rest[0]);
      final List<double> clsLogits = rest.sublist(1, 81);
      int argMax = 0;
      double best = -1e9;
      for (int i = 0; i < clsLogits.length; i++) {
        final double s = scoresAreProbabilities
            ? clsLogits[i]
            : sigmoid(clsLogits[i]);
        if (s > best) {
          best = s;
          argMax = i;
        }
      }
      scores.add(obj * best);
      clsIds.add(argMax);
      xywhs.add(xywh);
    }
  }

  final List<int> keep0 = <int>[];
  for (int i = 0; i < scores.length; i++) {
    if (scores[i] >= confThres) keep0.add(i);
  }
  if (keep0.isEmpty) return <Detection>[];

  final List<List<double>> keptXywh = [for (final int i in keep0) xywhs[i]];
  final List<int> keptCls = [for (final int i in keep0) clsIds[i]];
  final List<double> keptScore = [for (final int i in keep0) scores[i]];

  if (keptXywh.isNotEmpty && median([for (final v in keptXywh) v[2]]) <= 2.0) {
    for (final List<double> v in keptXywh) {
      v[0] *= inputWidth.toDouble();
      v[1] *= inputHeight.toDouble();
      v[2] *= inputWidth.toDouble();
      v[3] *= inputHeight.toDouble();
    }
  }

  final List<List<double>> boxesLtr = [
    for (final List<double> v in keptXywh) xywhToXyxy(v),
  ];
  final List<List<double>> boxes = <List<double>>[];
  for (final List<double> b in boxesLtr) {
    boxes.add(scaleFromLetterbox(b, r, dw, dh));
  }
  final double iw = imageWidth.toDouble();
  final double ih = imageHeight.toDouble();
  for (final List<double> b in boxes) {
    b[0] = b[0].clamp(0.0, iw);
    b[2] = b[2].clamp(0.0, iw);
    b[1] = b[1].clamp(0.0, ih);
    b[3] = b[3].clamp(0.0, ih);
  }

  final int effectiveTopk;
  if (topkPreNms > 0) {
    effectiveTopk = topkPreNms;
  } else {
    const int basePixels = 640 * 640;
    const int baseCandidates = 100;
    final int imagePixels = imageWidth * imageHeight;
    final double scale = imagePixels / basePixels;
    effectiveTopk = (baseCandidates * scale).round().clamp(20, 200);
  }

  if (effectiveTopk > 0 && keptScore.length > effectiveTopk) {
    final List<int> ord = argSortDesc(keptScore).take(effectiveTopk).toList();
    final List<List<double>> sortedBoxes = <List<double>>[];
    final List<double> sortedScores = <double>[];
    final List<int> sortedCls = <int>[];
    for (final int i in ord) {
      sortedBoxes.add(boxes[i]);
      sortedScores.add(keptScore[i]);
      sortedCls.add(keptCls[i]);
    }
    boxes
      ..clear()
      ..addAll(sortedBoxes);
    keptScore
      ..clear()
      ..addAll(sortedScores);
    keptCls
      ..clear()
      ..addAll(sortedCls);
  }

  if (filterClassId != null) {
    final List<List<double>> fBoxes = <List<double>>[];
    final List<double> fScores = <double>[];
    final List<int> fCls = <int>[];
    for (int i = 0; i < keptCls.length; i++) {
      if (keptCls[i] == filterClassId) {
        fBoxes.add(boxes[i]);
        fScores.add(keptScore[i]);
        fCls.add(keptCls[i]);
      }
    }
    boxes
      ..clear()
      ..addAll(fBoxes);
    keptScore
      ..clear()
      ..addAll(fScores);
    keptCls
      ..clear()
      ..addAll(fCls);
  }

  final List<int> keep = nms(
    boxes,
    keptScore,
    iouThres: iouThres,
    maxDet: maxDet,
  );
  final List<Detection> out = <Detection>[];
  for (final int i in keep) {
    out.add(
      Detection(cls: keptCls[i], score: keptScore[i], bboxXYXY: boxes[i]),
    );
  }
  return out;
}

/// Decodes a YOLOv8 detection head directly from the flat [out] Float32 buffer.
///
/// Fast-path twin of [postProcessDetections] for when the model output is
/// available as a flat [Float32List] (e.g. a CompiledModel output or a flat
/// interpreter output buffer). The generic path rebuilds the dense
/// `[1, channels, anchors]` tensor as nested boxed-double lists and transposes
/// it every call (~2x the dense output allocated as boxed doubles), which for
/// YOLOv8's `[1, 84, 8400]` head dominates per-frame cost.
///
/// Here the per-anchor candidate loop reads straight from [out] by index and
/// only the post-confidence-filter candidate set (small) uses lists. The same
/// score activation / letterbox / top-k / NMS helpers are reused, so results
/// are identical to [postProcessDetections].
///
/// [channelMajor] true means [out] is laid out as `[1, channels, anchors]`
/// (value at channel c, anchor a is `out[c * anchors + a]`, the YOLOv8 default);
/// false means `[1, anchors, channels]` (`out[a * channels + c]`).
///
/// [scoresAreProbabilities] skips sigmoid for models whose class/objectness
/// outputs already contain probabilities. It defaults to false for backward
/// compatibility with models that emit logits.
///
/// [useFastSingleClass] (default false) selects an EXPERIMENTAL fast path that,
/// when [filterClassId] is set, prunes each anchor on the filter-class channel
/// *before* the full argmax, skipping the ~80-class argmax for the ~99% of
/// anchors whose filter-class score is below [confThres]. It is ~50% faster at
/// decoding (measured ref p50 ~3.6ms -> fast ~1.7ms on -d macos, debug).
///
/// It is OFF by default because it is NOT result-equivalent and it LOWERS
/// PRECISION. The default path applies a cross-class top-k cap (keep the global
/// top [effectiveTopk] by score) *before* filtering to the target class; that
/// cap incidentally suppresses near-threshold filter-class false positives that
/// are out-scored by other classes. The fast path collects only filter-class
/// anchors, so it never sees that cross-class ranking and emits those FPs:
/// measured on a single-person frame at conf 0.5 it inflated 1 detection to
/// 4-10 (extra boxes all sitting at the confidence floor).
///
/// Computing the global top-k *requires* each anchor's max score, which is the
/// argmax this path skips, so there is no byte-identical way to avoid the cost.
/// Only enable this for recall-over-precision use cases, ideally paired with a
/// higher [confThres].
List<Detection> postProcessDetectionsFlat(
  Float32List out, {
  required int channels,
  required int anchors,
  required bool channelMajor,
  required int inputWidth,
  required int inputHeight,
  required double r,
  required int dw,
  required int dh,
  required int imageWidth,
  required int imageHeight,
  required double confThres,
  required double iouThres,
  required int maxDet,
  int? filterClassId,
  bool useFastSingleClass = false,
  bool scoresAreProbabilities = false,
}) {
  // Explicit up-front bound: the SIMD path reads through out.buffer with
  // absolute offsets, which would otherwise silently read past out's
  // logical extent (e.g. a view into a larger buffer) where the scalar
  // element reads threw.
  if (out.length < channels * anchors) {
    throw ArgumentError.value(
      out.length,
      'out.length',
      'Expected at least channels * anchors = ${channels * anchors} floats.',
    );
  }

  // C == 84 -> 4 box coords + 80 class scores (YOLOv8). Otherwise treat
  // channel 4 as objectness and the remainder as class logits (YOLOv5-style),
  // matching postProcessDetections.
  final bool hasObjectness = channels != 84;
  final int classStart = hasObjectness ? 5 : 4;
  final int numClasses = channels - classStart;

  final List<double> scores = <double>[];
  final List<int> clsIds = <int>[];
  final List<List<double>> xywhs = <List<double>>[];

  final bool fastSingleClass =
      useFastSingleClass &&
      filterClassId != null &&
      filterClassId >= 0 &&
      filterClassId < numClasses;

  // Any objectness factor is at most 1, so an anchor whose winning class value
  // is below the activation-specific threshold cannot pass the final score
  // check. Probability outputs compare directly; logit outputs compare against
  // the inverse-sigmoid threshold to avoid exp() for rejected anchors.
  final double classValueThres = scoresAreProbabilities
      ? confThres
      : confThres <= 0
      ? double.negativeInfinity
      : confThres >= 1
      ? double.infinity
      : math.log(confThres / (1 - confThres));

  double activateScore(double value) =>
      scoresAreProbabilities ? value : sigmoid(value);

  // Reads the 4 box coords for anchor [a] into the candidate lists.
  void addCandidate(int a, int cls, double score) {
    final double cx, cy, w, h;
    if (channelMajor) {
      cx = out[a];
      cy = out[anchors + a];
      w = out[2 * anchors + a];
      h = out[3 * anchors + a];
    } else {
      final int b = a * channels;
      cx = out[b];
      cy = out[b + 1];
      w = out[b + 2];
      h = out[b + 3];
    }
    scores.add(score);
    clsIds.add(cls);
    xywhs.add(<double>[cx, cy, w, h]);
  }

  double objScore(int a) => hasObjectness
      ? activateScore(
          channelMajor ? out[4 * anchors + a] : out[a * channels + 4],
        )
      : 1.0;

  if (fastSingleClass) {
    // Fast path: prune on the filter-class channel, argmax only on survivors.
    final int fc = filterClassId;
    for (int a = 0; a < anchors; a++) {
      final double fValue = channelMajor
          ? out[(classStart + fc) * anchors + a]
          : out[a * channels + classStart + fc];
      // Same score expression and threshold as the full path, computed from the
      // filter-class value. When the filter class is the winner (the only case
      // the full path keeps) fValue == bestValue, so this is identical; when it
      // is not the winner the full path would drop the anchor at the class
      // filter anyway, so pruning here is exact.
      if (fValue < classValueThres) continue;
      final double score = activateScore(fValue) * objScore(a);
      if (score < confThres) continue;

      // Confirm the filter class is the argmax (matches the full path's
      // cls == filterClassId filter); skip the rare non-winning survivors.
      int argMax = 0;
      double bestValue = -1e30;
      if (channelMajor) {
        int idx = classStart * anchors + a;
        for (int k = 0; k < numClasses; k++) {
          final double v = out[idx];
          if (v > bestValue) {
            bestValue = v;
            argMax = k;
          }
          idx += anchors;
        }
      } else {
        final int base = a * channels + classStart;
        for (int k = 0; k < numClasses; k++) {
          final double v = out[base + k];
          if (v > bestValue) {
            bestValue = v;
            argMax = k;
          }
        }
      }
      if (argMax != fc) continue;
      addCandidate(a, fc, score);
    }
  } else if (channelMajor && anchors % 4 == 0 && out.offsetInBytes % 16 == 0) {
    // Channel-major SIMD: each class plane is a contiguous run of `anchors`
    // floats, so sweep the planes with Float32x4 to carry the running max
    // logit per anchor in vector lanes (lanes are independent anchors).
    // greaterThan is strict, so the first class wins ties exactly like the
    // scalar loop. The winning class index is recovered with a scalar argmax
    // below, only for the anchors that clear the threshold; see the comment
    // there for why the argmax is not carried in SIMD lanes.
    final int vecCount = anchors ~/ 4;
    final Float32x4List bestValues4 = Float32x4List(vecCount);
    final Float32x4 minusInf = Float32x4.splat(-1e30);
    for (int i = 0; i < vecCount; i++) {
      bestValues4[i] = minusInf;
    }
    for (int k = 0; k < numClasses; k++) {
      final Float32x4List plane = Float32x4List.view(
        out.buffer,
        out.offsetInBytes + (classStart + k) * anchors * 4,
        vecCount,
      );
      for (int i = 0; i < vecCount; i++) {
        final Float32x4 v = plane[i];
        final Float32x4 best = bestValues4[i];
        bestValues4[i] = v.greaterThan(best).select(v, best);
      }
    }
    final Float32List bestFlat = Float32List.view(bestValues4.buffer);
    // Recover the winning class with a scalar argmax over the (few) anchors that
    // clear the threshold. The earlier SIMD variant carried the argmax in
    // Float32x4 lanes via greaterThan().select(); the Dart ARM64 JIT
    // miscompiles that carry (nondeterministic across optimization tiers,
    // yielding a wrong argmax that invents phantom detections), so keep only the
    // max reduction in SIMD and argmax scalar per survivor.
    for (int a = 0; a < anchors; a++) {
      final double bestValue = bestFlat[a];
      if (bestValue < classValueThres) continue;
      final double score = activateScore(bestValue) * objScore(a);
      if (score < confThres) continue;
      int argMax = 0;
      double argBest = -1e30;
      int idx = classStart * anchors + a;
      for (int k = 0; k < numClasses; k++) {
        final double v = out[idx];
        if (v > argBest) {
          argBest = v;
          argMax = k;
        }
        idx += anchors;
      }
      addCandidate(a, argMax, score);
    }
  } else {
    // Reference path: full argmax over every anchor, no pre-filter.
    // The configured activation is monotonic, so the class argmax can operate
    // on raw values and activation happens once per surviving anchor.
    for (int a = 0; a < anchors; a++) {
      int argMax = 0;
      double bestValue = -1e30;
      if (channelMajor) {
        int idx = classStart * anchors + a;
        for (int k = 0; k < numClasses; k++) {
          final double v = out[idx];
          if (v > bestValue) {
            bestValue = v;
            argMax = k;
          }
          idx += anchors;
        }
      } else {
        final int base = a * channels + classStart;
        for (int k = 0; k < numClasses; k++) {
          final double v = out[base + k];
          if (v > bestValue) {
            bestValue = v;
            argMax = k;
          }
        }
      }

      if (bestValue < classValueThres) continue;
      final double score = activateScore(bestValue) * objScore(a);
      if (score < confThres) continue;
      addCandidate(a, argMax, score);
    }
  }

  if (scores.isEmpty) return <Detection>[];

  // Same normalized-vs-pixel coordinate guard as postProcessDetections.
  if (median(<double>[for (final v in xywhs) v[2]]) <= 2.0) {
    for (final List<double> v in xywhs) {
      v[0] *= inputWidth;
      v[1] *= inputHeight;
      v[2] *= inputWidth;
      v[3] *= inputHeight;
    }
  }

  final double iw = imageWidth.toDouble();
  final double ih = imageHeight.toDouble();
  List<List<double>> boxes = <List<double>>[];
  for (final List<double> v in xywhs) {
    final List<double> b = scaleFromLetterbox(xywhToXyxy(v), r, dw, dh);
    b[0] = b[0].clamp(0.0, iw);
    b[2] = b[2].clamp(0.0, iw);
    b[1] = b[1].clamp(0.0, ih);
    b[3] = b[3].clamp(0.0, ih);
    boxes.add(b);
  }

  // Auto top-k pre-NMS, identical to postProcessDetections (topkPreNms == 0).
  const int basePixels = 640 * 640;
  final double scaleFactor = (imageWidth * imageHeight) / basePixels;
  final int effectiveTopk = (100 * scaleFactor).round().clamp(20, 200);

  List<double> sc = scores;
  List<int> cl = clsIds;
  if (sc.length > effectiveTopk) {
    final List<int> ord = argSortDesc(sc).take(effectiveTopk).toList();
    boxes = <List<double>>[for (final int i in ord) boxes[i]];
    sc = <double>[for (final int i in ord) scores[i]];
    cl = <int>[for (final int i in ord) clsIds[i]];
  }

  if (filterClassId != null) {
    final List<List<double>> fb = <List<double>>[];
    final List<double> fs = <double>[];
    final List<int> fc = <int>[];
    for (int i = 0; i < cl.length; i++) {
      if (cl[i] == filterClassId) {
        fb.add(boxes[i]);
        fs.add(sc[i]);
        fc.add(cl[i]);
      }
    }
    boxes = fb;
    sc = fs;
    cl = fc;
  }

  final List<int> keep = nms(boxes, sc, iouThres: iouThres, maxDet: maxDet);
  return <Detection>[
    for (final int i in keep)
      Detection(cls: cl[i], score: sc[i], bboxXYXY: boxes[i]),
  ];
}

/// Transposes a 2D list (swaps rows and columns).
List<List<double>> transpose2D(List<List<double>> a) {
  if (a.isEmpty) return <List<double>>[];
  final int rows = a.length, cols = a[0].length;
  final List<List<double>> out = List.generate(
    cols,
    (_) => List<double>.filled(rows, 0.0),
  );
  for (int r = 0; r < rows; r++) {
    final List<double> row = a[r];
    for (int c = 0; c < cols; c++) {
      out[c][r] = row[c];
    }
  }
  return out;
}

/// Concatenates a list of 2D matrices along axis 0.
List<List<double>> concat0(List<List<List<double>>> parts) {
  final List<List<double>> out = <List<double>>[];
  for (final List<List<double>> p in parts) {
    out.addAll(p);
  }
  return out;
}

/// Ensures a dynamic list is a proper 2D `List<List<double>>`.
List<List<double>> ensure2D(List<dynamic> raw) {
  return raw
      .map<List<double>>(
        (e) => (e as List).map((v) => (v as num).toDouble()).toList(),
      )
      .toList();
}

/// Converts XYWH bounding box format to XYXY format.
List<double> xywhToXyxy(List<double> xywh) {
  final double cx = xywh[0], cy = xywh[1], w = xywh[2], h = xywh[3];
  return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
}

/// Decodes raw detection model outputs into a standardized 2D matrix.
/// Handles both `[1, numBoxes, 5 + classes]` and
/// `[1, 5 + classes, numBoxes]` formats.
List<List<double>> decodeDetectionOutputs(List<dynamic> outputs) {
  final List<List<List<double>>> parts = <List<List<double>>>[];
  for (final raw in outputs) {
    final List<dynamic> t3d = raw as List;
    if (t3d.length != 1) throw StateError('Unexpected output rank');

    final List<List<double>> out2d = ensure2D(t3d[0]);
    if (out2d.isEmpty) continue;

    final int rows = out2d.length;
    final int cols = out2d[0].length;
    if (rows < cols && (rows == 84 || rows == 85)) {
      parts.add(transpose2D(out2d));
    } else {
      parts.add(out2d);
    }
  }

  final List<List<double>> out = concat0(parts);
  if (out.isEmpty || out[0].length < 84) {
    throw StateError('Expected channels >=84');
  }
  return out;
}
