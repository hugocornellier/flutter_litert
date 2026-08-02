#!/usr/bin/env python3
"""Cross-checks flutter_litert against the official LiteRT Python API on macOS.

The Flutter matrix measures every backend against its own plain-CPU Dart
reference. That detects backend-versus-backend divergence but is blind to a
fault in the binding layer itself, because a wrong baseline would make every
backend agree with it. Google's `ai-edge-litert` wheel loads the same
`libLiteRt` and the same Metal accelerator through an independent
implementation, so it can act as an external oracle for two questions the
Flutter dataset cannot answer alone:

1. Do Python and Dart produce the same CPU outputs for identical inputs? A
   mismatch points at flutter_litert's marshalling rather than at a backend.
2. Is CompiledModel GPU fp16's poor parity an upstream LiteRT property or an
   artefact of our bindings? `GpuOptions.enforce_f32` is the same switch the
   Dart side sets, so the fp16-versus-fp32 split is directly reproducible.

Coverage stops where Python's surface does: the Core ML delegate, the Core ML
NPU accelerator, and Flex are Objective-C APIs with no Python binding, so those
Flutter columns have no counterpart here.

Fixtures, tolerance, and the tensor summary are deliberate transliterations of
`example/integration_test/apple_model_matrix_test.dart`. They must stay in step
with it or the comparison silently stops meaning anything.

Usage:
  python3 test/benchmark/python_litert_crosscheck.py \
      --model-root ~/IdeaProjects \
      --out test/benchmark/PYTHON_LITERT_CROSSCHECK.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback

import numpy as np

# Mirrors _absoluteTolerance / _relativeTolerance in the Dart harness.
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 0.01
REQUIRED_FIXTURES = 3
FIXTURE_NAMES = [
    "constant_0_5",
    "ramp_0_05_0_95",
    "scrambled_0_1_0_9",
    "reverse_ramp_0_05_0_95",
    "scrambled_0_2_0_8",
]

# The same 29 published models the Flutter matrices pin.
MODELS = [
    ("face_detection_tflite", "face_blendshapes.tflite"),
    ("face_detection_tflite", "face_detection_back.tflite"),
    ("face_detection_tflite", "face_detection_front.tflite"),
    ("face_detection_tflite", "face_detection_full_range.tflite"),
    ("face_detection_tflite", "face_detection_full_range_sparse.tflite"),
    ("face_detection_tflite", "face_detection_short_range.tflite"),
    ("face_detection_tflite", "face_landmark.tflite"),
    ("face_detection_tflite", "iris_landmark.tflite"),
    ("face_detection_tflite", "mobilefacenet.tflite"),
    ("face_detection_tflite", "selfie_multiclass.tflite"),
    ("face_detection_tflite", "selfie_segmenter.tflite"),
    ("face_detection_tflite", "selfie_segmenter_landscape.tflite"),
    ("pose_detection", "pose_landmark_full.tflite"),
    ("pose_detection", "pose_landmark_heavy.tflite"),
    ("pose_detection", "pose_landmark_lite.tflite"),
    ("pose_detection", "yolov8n_float32.tflite"),
    ("hand_detection", "canned_gesture_classifier.tflite"),
    ("hand_detection", "gesture_embedder.tflite"),
    ("hand_detection", "hand_detection.tflite"),
    ("hand_detection", "hand_landmark_full.tflite"),
    ("animal_detection", "species_classifier_float16.tflite"),
    ("animal_detection", "superanimal_rtmpose_s_float16.tflite"),
    ("animal_detection", "superanimal_ssdlite_float16.tflite"),
    ("cat_detection", "cat_face_landmarks_full.tflite"),
    ("cat_detection", "cat_face_localizer.tflite"),
    ("dog_detection", "dog_face_landmarks_full.tflite"),
    ("dog_detection", "dog_face_localizer.tflite"),
    ("object_detection", "efficientdet_lite0.tflite"),
    ("object_detection", "efficientdet_lite2.tflite"),
]


def make_input(values: int, tensor_index: int, fixture_index: int) -> np.ndarray:
    """Transliteration of _makeInput; index arithmetic must match exactly."""
    i = np.arange(values, dtype=np.int64)
    if fixture_index == 0:
        out = np.full(values, 0.5, dtype=np.float64)
    elif fixture_index == 1:
        out = 0.05 + 0.9 * ((i + tensor_index * 17) % 251) / 250.0
    elif fixture_index == 2:
        out = 0.1 + 0.8 * ((i * 73 + tensor_index * 31) % 251) / 250.0
    elif fixture_index == 3:
        out = 0.95 - 0.9 * ((i + tensor_index * 19) % 251) / 250.0
    elif fixture_index == 4:
        out = 0.2 + 0.6 * ((i * 101 + tensor_index * 47) % 251) / 250.0
    else:
        raise IndexError(f"fixture_index {fixture_index} out of range")
    return out.astype(np.float32)


def summarize(tensor: np.ndarray) -> dict:
    """Transliteration of _TensorSummary.from, including the weighted checksum.

    The checksum weights each element by 1 + (i % 97) over the flattened
    tensor, so it is sensitive to ordering and therefore to a layout mismatch
    that min/max/mean would hide.
    """
    flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]
    n_finite = int(finite.size)
    result = {
        "values": int(flat.size),
        "finite_values": n_finite,
        "non_finite_values": int(flat.size - n_finite),
    }
    if n_finite == 0:
        result.update(
            min=None, max=None, mean=None, rms=None,
            weighted_checksum=None, top_index=-1, top_value=None,
        )
        return result
    idx = np.arange(flat.size, dtype=np.int64)
    weights = (1 + (idx % 97)).astype(np.float64)
    checksum = float(np.sum(flat[finite_mask].astype(np.float64)
                            * weights[finite_mask]))
    # Dart tracks the first strictly-greater element, which is argmax's
    # first-occurrence tie-break over the finite subset.
    finite_positions = idx[finite_mask]
    top_pos = int(finite_positions[int(np.argmax(finite))])
    result.update(
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.sum(finite.astype(np.float64)) / n_finite),
        rms=float(math.sqrt(np.sum(finite.astype(np.float64) ** 2) / n_finite)),
        weighted_checksum=checksum,
        top_index=top_pos,
        top_value=float(flat[top_pos]),
    )
    return result


def compare(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict:
    """Applies the Dart tolerance: abs <= 1e-4 + 0.01 * max(range, magnitude)."""
    if len(reference) != len(candidate):
        return {"passed": False, "reason": "output_count_mismatch"}
    worst_abs = 0.0
    worst_ratio = 0.0
    worst_rel = 0.0
    for ref, cand in zip(reference, candidate):
        r = np.asarray(ref, dtype=np.float64).reshape(-1)
        c = np.asarray(cand, dtype=np.float64).reshape(-1)
        if r.size != c.size:
            return {"passed": False, "reason": "value_count_mismatch"}
        if not np.all(np.isfinite(c)):
            return {"passed": False, "reason": "non_finite_output"}
        scale = max(float(np.max(r) - np.min(r)) if r.size else 0.0,
                    float(np.max(np.abs(r))) if r.size else 0.0)
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * scale
        diff = np.abs(r - c)
        worst_abs = max(worst_abs, float(np.max(diff)) if diff.size else 0.0)
        if tolerance > 0 and diff.size:
            worst_ratio = max(worst_ratio, float(np.max(diff)) / tolerance)
        denom = np.maximum(np.abs(r), 1e-12)
        if diff.size:
            worst_rel = max(worst_rel, float(np.max(diff / denom)))
    return {
        "passed": worst_ratio <= 1.0,
        "worst_absolute_error": worst_abs,
        "worst_relative_error": worst_rel,
        "worst_tolerance_ratio": worst_ratio,
        "reason": None,
    }


def run_interpreter(path: str, inputs_for, num_threads: int,
                    xnnpack: bool = True) -> tuple:
    """Runs the classic Interpreter and returns (outputs_per_fixture, timing).

    LiteRT's Python Interpreter applies XNNPACK to float models by default, so
    "CPU" here would silently mean XNNPACK unless default delegates are turned
    off. BUILTIN_WITHOUT_DEFAULT_DELEGATES gives the reference kernels, which
    is what the Dart harness's plain-CPU mode uses as its accuracy baseline.
    """
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    interp = Interpreter(
        model_path=path,
        num_threads=num_threads,
        experimental_op_resolver_type=(
            OpResolverType.AUTO if xnnpack
            else OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    per_fixture = []
    for fixture_index in inputs_for:
        for slot, detail in enumerate(in_details):
            values = int(np.prod(detail["shape"]))
            data = make_input(values, slot, fixture_index)
            interp.set_tensor(detail["index"],
                              data.reshape(detail["shape"]).astype(detail["dtype"]))
        interp.invoke()
        per_fixture.append(
            [np.array(interp.get_tensor(d["index"])) for d in out_details]
        )
    # Warm then time the invoke only, matching the Dart timing scope.
    for _ in range(5):
        interp.invoke()
    samples = []
    for _ in range(15):
        t0 = time.perf_counter()
        interp.invoke()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return per_fixture, samples


def _read_buffer(buffer) -> np.ndarray:
    """Reads a TensorBuffer, which needs an explicit element count and dtype."""
    details = buffer.get_tensor_details()
    shape = tuple(int(x) for x in details["shape"])
    count = int(np.prod(shape)) if shape else 0
    values = buffer.read(count, np.float32)
    return np.asarray(values, dtype=np.float32).reshape(shape)


def run_compiled(path: str, inputs_for, accel: str, enforce_f32: bool) -> tuple:
    """Runs CompiledModel on the requested accelerator set.

    The accelerator lives inside Options, and the API rejects passing both
    Options and hardware_accel, so everything goes through Options here.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.gpu_options import GpuOptions
    from ai_edge_litert.options import Options

    flags = {"cpu": HardwareAccelerator.CPU, "gpu": HardwareAccelerator.GPU}
    hw = None
    for token in accel.split("+"):
        hw = flags[token] if hw is None else hw | flags[token]

    options = Options(
        hardware_accelerators=hw,
        gpu_options=GpuOptions(enforce_f32=True) if enforce_f32 else None,
    )
    model = CompiledModel.from_file(path, options=options)

    signature_index = 0
    signature_key = None
    try:
        signature_key = model.get_signature_by_index(signature_index).get("key")
    except Exception:
        pass
    if signature_key is None:
        keys = list(model.get_signature_list().keys())
        signature_key = keys[0] if keys else ""

    in_details = model.get_input_tensor_details(signature_key)
    in_buffers = model.create_input_buffers(signature_index)
    out_buffers = model.create_output_buffers(signature_index)

    # Details come back keyed by tensor name; order them to match the buffers.
    ordered_inputs = list(in_details.values()) if isinstance(in_details, dict) \
        else list(in_details)

    per_fixture = []
    for fixture_index in inputs_for:
        for slot, buffer in enumerate(in_buffers):
            detail = ordered_inputs[slot] if slot < len(ordered_inputs) else None
            shape = detail["shape"] if detail and "shape" in detail else None
            if shape is None:
                shape = buffer.get_tensor_details()["shape"]
            values = int(np.prod(shape))
            data = make_input(values, slot, fixture_index)
            buffer.write(data.reshape(tuple(int(x) for x in shape)))
        model.run_by_index(signature_index, in_buffers, out_buffers)
        per_fixture.append([_read_buffer(b) for b in out_buffers])

    for _ in range(5):
        model.run_by_index(signature_index, in_buffers, out_buffers)
    samples = []
    for _ in range(15):
        t0 = time.perf_counter()
        model.run_by_index(signature_index, in_buffers, out_buffers)
        samples.append((time.perf_counter() - t0) * 1000.0)

    accelerated = None
    try:
        accelerated = bool(model.is_fully_accelerated())
    except Exception:
        pass
    return per_fixture, samples, accelerated


MODES = [
    ("interpreter_cpu_4t", "interpreter", dict(num_threads=4, xnnpack=False)),
    ("interpreter_xnnpack_4t", "interpreter", dict(num_threads=4, xnnpack=True)),
    ("compiled_cpu_fp32", "compiled", dict(accel="cpu", enforce_f32=True)),
    ("compiled_gpu_fp16", "compiled", dict(accel="gpu", enforce_f32=False)),
    ("compiled_gpu_fp32", "compiled", dict(accel="gpu", enforce_f32=True)),
    ("compiled_gpu_cpu_fp32", "compiled", dict(accel="gpu+cpu", enforce_f32=True)),
]


def compare_with_flutter(payload: dict, flutter_path: str,
                         max_deviation: float) -> int:
    """Reports agreement and fails if the CPU baselines have drifted.

    Verdict differences are expected and interesting: they are what the oracle
    exists to surface. A drifting CPU *reference* is different in kind, because
    it means the two harnesses stopped feeding the same inputs and every other
    comparison silently lost its meaning.
    """
    with open(flutter_path) as handle:
        flutter = json.load(handle)

    flutter_rows = {
        (r["repository"] + "/" + r["model_name"], r["mode"]): r
        for r in flutter["rows"]
    }
    flutter_refs = {
        r["model_name"]: r for r in flutter["references"]
        if r.get("status") == "ok"
    }

    def verdict(row):
        if row is None or row.get("status") != "ok":
            return "not-run"
        return "pass" if row.get("accuracy_pass") else "FAIL"

    agree, differ = 0, []
    for row in payload["rows"]:
        other = flutter_rows.get((row["model"], row["mode"]))
        if verdict(row) == verdict(other):
            agree += 1
        else:
            differ.append((row["model"], row["mode"],
                           verdict(other), verdict(row)))

    worst_deviation, worst_model, compared = 0.0, None, 0
    mismatched_fixtures = []
    # Models that needed the custom-op fallback have an XNNPACK baseline here
    # against reference kernels in Dart. That is a recorded difference in what
    # was measured, not drift in how it was measured, so gating on it would
    # only teach us to ignore the gate.
    not_baseline_comparable = []
    for ref in payload["references"]:
        if ref.get("status") != "ok":
            continue
        if ref.get("reference_backend") != "builtin_only":
            not_baseline_comparable.append(ref["model"].split("/")[-1])
            continue
        other = flutter_refs.get(ref["model"].split("/")[-1])
        if other is None:
            continue
        if ref["fixtures"] != [f["name"] for f in other["fixtures"]]:
            mismatched_fixtures.append(ref["model"])
            continue
        for mine, theirs in zip(ref["fixture_summaries"], other["fixtures"]):
            for a, b in zip(mine, theirs["outputs"]):
                x, y = a.get("weighted_checksum"), b.get("weighted_checksum")
                if x is None or y is None:
                    continue
                compared += 1
                deviation = abs(x - y) / max(abs(x), abs(y), 1e-9)
                if deviation > worst_deviation:
                    worst_deviation, worst_model = deviation, ref["model"]

    print(f"\ncells compared: {agree + len(differ)}  agree: {agree}  "
          f"differ: {len(differ)}")
    for model, mode, theirs, mine in differ:
        print(f"   {model:<52}{mode:<24}flutter={theirs:<9}python={mine}")
    print(f"CPU reference outputs compared: {compared}")
    if not_baseline_comparable:
        print("excluded from the reference gate (XNNPACK fallback baseline): "
              f"{not_baseline_comparable}")
    print(f"worst relative checksum deviation: {worst_deviation:.3e}"
          f" ({worst_model})")

    failed = False
    if mismatched_fixtures:
        print(f"FAIL: fixture sets diverged for {mismatched_fixtures}")
        failed = True
    if compared and worst_deviation > max_deviation:
        print(f"FAIL: CPU references drifted beyond {max_deviation:.1e};"
              " the transliterated fixtures or tolerance are out of step")
        failed = True
    if not failed:
        print("OK: CPU references agree within float32 noise")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-filter", default="")
    ap.add_argument("--mode-filter", default="")
    ap.add_argument(
        "--compare",
        metavar="FLUTTER_JSON",
        help="After collecting, compare against a Flutter matrix dataset and "
             "exit nonzero if the CPU references have drifted apart. This is "
             "the guard against the transliterated fixtures silently diverging "
             "from the Dart harness.",
    )
    ap.add_argument("--max-reference-deviation", type=float, default=1e-4)
    args = ap.parse_args()

    root = os.path.expanduser(args.model_root)
    selected_models = [
        (repo, f) for repo, f in MODELS
        if not args.model_filter
        or f"{repo}/{f[:-len('.tflite')]}" in args.model_filter.split(",")
    ]
    selected_modes = [
        m for m in MODES
        if not args.mode_filter or m[0] in args.mode_filter.split(",")
    ]

    rows, references = [], []
    for repo, filename in selected_models:
        label = f"{repo}/{filename[: -len('.tflite')]}"
        path = os.path.join(root, repo, "assets", "models", filename)
        if not os.path.exists(path):
            for mode_name, _, _ in selected_modes:
                rows.append({"model": label, "mode": mode_name,
                             "status": "model_missing", "error": path})
            continue

        # Pick the first REQUIRED_FIXTURES candidates whose CPU reference is
        # finite, exactly as the Dart harness does.
        reference_outputs, chosen, rejected = None, [], []
        # Python couples custom-op registration and default delegates into one
        # resolver setting. Builtin-only gives the reference kernels the Dart
        # baseline uses, but cannot prepare MediaPipe custom ops such as
        # Convolution2DTransposeBias, which flutter_litert registers through
        # libtflite_custom_ops. Fall back for exactly those models and record
        # it, so a row's baseline is never silently XNNPACK.
        reference_xnnpack = False
        try:
            run_interpreter(path, [0], num_threads=4, xnnpack=False)
        except Exception as probe_error:
            if "custom op" in str(probe_error).lower():
                reference_xnnpack = True
            # Any other failure is reported by the real attempt below.
        try:
            for candidate in range(len(FIXTURE_NAMES)):
                if len(chosen) == REQUIRED_FIXTURES:
                    break
                outs, _ = run_interpreter(path, [candidate], num_threads=4,
                                          xnnpack=reference_xnnpack)
                if all(np.all(np.isfinite(o)) for o in outs[0]):
                    chosen.append(candidate)
                else:
                    rejected.append(FIXTURE_NAMES[candidate])
            if len(chosen) < REQUIRED_FIXTURES:
                raise RuntimeError("insufficient finite fixtures")
            reference_outputs, _ = run_interpreter(path, chosen,
                                                  num_threads=4,
                                                  xnnpack=reference_xnnpack)
        except Exception as error:
            for mode_name, _, _ in selected_modes:
                rows.append({"model": label, "mode": mode_name,
                             "status": "reference_failed", "error": str(error)})
            references.append({"model": label, "status": "reference_failed",
                               "error": str(error)})
            continue

        references.append({
            "model": label,
            "status": "ok",
            "reference_backend": ("builtin_with_default_delegates"
                                  if reference_xnnpack else "builtin_only"),
            "fixtures": [FIXTURE_NAMES[c] for c in chosen],
            "rejected_fixtures": rejected,
            # Summaries let this be joined against the Flutter reference block,
            # which stores the same statistics rather than raw tensors.
            "fixture_summaries": [
                [summarize(o) for o in outs] for outs in reference_outputs
            ],
        })

        for mode_name, kind, kwargs in selected_modes:
            row = {"model": label, "mode": mode_name}
            try:
                if kind == "interpreter":
                    call = dict(kwargs)
                    if reference_xnnpack and not call.get("xnnpack", True):
                        call["xnnpack"] = True
                    outs, samples = run_interpreter(path, chosen, **call)
                    accelerated = None
                else:
                    outs, samples, accelerated = run_compiled(path, chosen, **kwargs)
                cases = [compare(reference_outputs[i], outs[i])
                         for i in range(len(chosen))]
                row.update(
                    status="ok",
                    accuracy_pass=all(c["passed"] for c in cases),
                    accuracy_cases=cases,
                    worst_tolerance_ratio=max(
                        (c.get("worst_tolerance_ratio") or 0.0) for c in cases),
                    fully_accelerated=accelerated,
                    p50_ms=round(statistics.median(samples), 4),
                    summaries=[[summarize(o) for o in outs[i]]
                               for i in range(len(chosen))],
                )
            except Exception as error:
                row.update(status="error", error=str(error),
                           error_type=type(error).__name__,
                           traceback=traceback.format_exc(limit=3))
            rows.append(row)
            print(f"{label:<52} {mode_name:<24} {row['status']}"
                  f" acc={row.get('accuracy_pass')} p50={row.get('p50_ms')}",
                  flush=True)

    payload = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": "macos-python",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "litert_python_version": __import__("ai_edge_litert").__version__
            if hasattr(__import__("ai_edge_litert"), "__version__") else "unknown",
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "accuracy_kind": "cpu_reference_tensor_parity",
            "model_count": len(selected_models),
            "mode_count": len(selected_modes),
        },
        "references": references,
        "rows": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nwrote {len(rows)} rows ({ok} ok) to {args.out}")
    if args.compare:
        return compare_with_flutter(payload, args.compare,
                                    args.max_reference_deviation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
