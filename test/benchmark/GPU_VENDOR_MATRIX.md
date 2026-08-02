# CompiledModel GPU across four architectures

Collected 2026-08-03 from the 29 published models, run through the same modes,
fixtures, and tolerance on every platform. Sources:

- [macOS](MACOS_MODEL_MATRIX_RESULTS.json), Mac16,5 / Apple Metal, profile
- [Python](PYTHON_LITERT_CROSSCHECK.json), same Mac, Google's `ai-edge-litert`
- [Galaxy S23](ANDROID_MODEL_MATRIX_RESULTS.json), Adreno 740, debug
- [Pixel 9 Pro](ANDROID_PIXEL9PRO_MODEL_MATRIX_RESULTS.json), Mali-G715, debug
- [Galaxy A56](ANDROID_GALAXYA56_MODEL_MATRIX_RESULTS.json), Xclipse (RDNA), debug

Accuracy is deterministic CPU-reference tensor parity, which is unaffected by
the debug/profile distinction because the native libraries do the arithmetic
either way. Timings are not comparable between the profile and debug rows.

## Precision

| Platform | GPU | fp16 ran/pass | fp32 ran/pass |
|---|---|---:|---:|
| Galaxy A56 | Xclipse (RDNA) | 18 / 5 | **18 / 18** |
| Galaxy S23 | Adreno 740 | 18 / 4 | **18 / 18** |
| Pixel 9 Pro | Mali-G715 | 12 / 1 | **12 / 12** |
| Mac M4 | Apple Metal | 18 / 4 | **18 / 18** |
| Mac M4, Python | Apple Metal | 18 / 4 | **18 / 18** |

fp32 passed everything that ran, on four GPU architectures and through two
independent implementations. Mali's denominator of 12 rather than 18 is a
measurement artefact discussed below and does not weaken the precision result:
every model that ran passed at fp32 there too. fp16 failed the majority everywhere, and worst of
all on Mali. Google's own Python API reproduces the Apple numbers exactly
through the same `GpuOptions.enforce_f32` switch the Dart side sets, so this is
an upstream numerical property rather than a binding artefact.

The cause is not exotic: fp16 carries about three decimal digits of mantissa,
and these models emit pixel-space coordinates and landmark positions.
`CompiledModel.fromBufferWithGpuFallback` already defaults to fp32 for exactly
this reason; the plain constructors do not.

## Reachability, which is the real constraint

Every device fails the same 11 models in the same two ways:

| Failure | Count | Adreno | Xclipse | Mali |
|---|---:|:-:|:-:|:-:|
| `ok` | | 18 | 18 | 12* |
| `kLiteRtStatusErrorCompilation` (504) | 5 | yes | yes | yes |
| `LiteRtCreateCompiledModel` runtime failure | 6 | yes | yes | yes |
| `LiteRtLockTensorBuffer` runtime failure | 6 | no | no | yes* |

Those 11 models genuinely cannot run on a GPU today, consistently across all
three vendors. Accuracy is not what limits GPU adoption, reachability is.

\* Mali's 12 is not a coverage figure. See below.

## Mali's six extra failures are ordering-dependent, not model-specific

An earlier revision of this file read Mali's six extra failures as a
vendor-specific buffer-lock defect and concluded that Mali reaches only 12
models. That conclusion does not survive its own data.

`face_detection_front` and `face_detection_short_range` are the same file:
SHA-256 `3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633` for
both. In the Pixel run one passed and the other failed. No property of a model
can explain a byte-identical pair disagreeing.

The failures are positional. In the 29-model sequence, every lock failure sits
at index 19 or later and none appear before it:

```text
0..18   ok / 504 / runtime-failure     no lock failures
19      face_detection_full_range      LOCK
21      canned_gesture_classifier      LOCK
24      pose_landmark_heavy            LOCK
26      face_detection_back            LOCK
27      iris_landmark                  LOCK
28      face_detection_short_range     LOCK
```

The identical model passed at index 9 and failed at index 28. That is the
signature of cumulative resource exhaustion across a long-lived process, not of
an incompatible graph.

This also makes the earlier Adreno comparison unequal. The Galaxy S23 dataset
was collected in five shards of roughly six models per process, while the Pixel
and A56 ran all 29 in a single process. A per-model leak is invisible at six
models per process and only emerges in a 29-model one. The A56 showed no lock
failures under the same single-execution shape, so the leak is not purely a
function of run length either; both a long run and something Mali-specific
appear to be required.

The connection previously drawn to the open Windows `CompiledModel` lock flake
is weaker than it looked: `LiteRtStatus=3` is `kLiteRtStatusErrorRuntimeFailure`,
a catch-all, so sharing it proves little on its own. It becomes interesting
again only if the Windows case also turns out to be exhaustion rather than
model-specific.

The open question is therefore the leak, not a coverage limit. If confirmed, it
matters more than a missing-model list would: a long-running application would
degrade over time rather than simply declining to accelerate certain graphs.

## How this conclusion was reached, and three wrong turns

Recorded because the corrections are more instructive than the result, and
because each wrong turn came from the same mistake.

**First claim: "Mali reaches only 12 of 29 models, a vendor coverage limit."**
Wrong. It compared a Pixel run of 29 models in one process against a Galaxy S23
run of five shards at roughly six models each. Run length differed along with
the vendor.

**Second claim: "a Mali driver defect, so it affects most Pixels and MediaTek
phones."** Wrong. A Galaxy A35, Exynos 1380 with Mali-G68, produced a failure
profile byte-identical to Adreno and Xclipse: 18 ok, 5 compilation rejections,
6 runtime failures, zero lock failures. Mali as a family is not implicated.

**Third claim: "then it is specific to Google Tensor."** Also unsupported. The
same Pixel 9 Pro, run at 116 cells instead of 261, was completely clean. The
device is not the variable on its own.

What survived every control is the threshold. Identical hardware, identical
models, identical GPU modes: 261 cells fails, 116 cells does not. Two facts
made it undeniable. `face_detection_front` and `face_detection_short_range` are
the same file, SHA-256 `3bc182eb9f33925d…`, and the long run passed one and
failed the other. And every lock failure sat at index 19 or later, none before.

The common error each time was reading a vendor difference out of runs that
also differed in length. Comparisons here should hold process shape fixed and
vary one thing.

A macOS probe narrows it further: 300 strict-GPU compiles in one process, both
through these bindings and through Google's `ai-edge-litert` Python API, with
no failures on Metal. The shared Dart and FFI layer is therefore not leaking,
since it is identical on every platform and would fail on Metal too. That
points at the Android GPU accelerator, which is upstream. It is narrowing
rather than proof: Python cannot practically run on Android, so the comparison
that settles it for Metal has no equivalent there.

## What this supports

- Default CompiledModel to fp32. The evidence spans four GPU architectures and
  two implementations, and fp32 has no accuracy counterexample anywhere.
- Treat fp16 as an explicit, per-model opt-in validated on the target GPU.
- Build any GPU allowlist from reachability, not precision. Do not treat Mali's
  12 as its ceiling until the exhaustion question is settled.
- Investigate the Mali lock failures as a per-model resource leak. The
  distinguishing test is a short run over only the affected models: if they
  pass there, the coverage gap is a measurement artefact.
- Collect future device datasets with a comparable process shape. Mixing
  five-shard and single-execution runs makes per-process effects look like
  vendor differences.
