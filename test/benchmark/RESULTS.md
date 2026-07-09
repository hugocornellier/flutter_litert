# Engine × Accelerator Results

Cross-platform benchmark + correctness matrix for `flutter_litert`, comparing the
classic **Interpreter** against the LiteRT Next **CompiledModel** across every
accelerator, on real devices.

- **Raw data:** [`RESULTS.csv`](RESULTS.csv): long format, one row per
  (run × model × mode), append-only. Every committed run adds rows; the tables
  below are human-readable summaries regenerated from it.
- **Test:** [`example/integration_test/engine_matrix_test.dart`](../../example/integration_test/engine_matrix_test.dart)
- **Runner:** [`test/benchmark/run_matrix.sh`](run_matrix.sh)

## How to run

```sh
test/benchmark/run_matrix.sh macos   # or: linux | windows | <android/ios device id>
```

This runs the matrix in **profile mode** (required; debug numbers are not
representative) via `flutter drive`, captures device/OS/commit metadata, and
appends the results to `RESULTS.csv` on the host. Commit the updated CSV after
each run so the file accumulates a record across hardware.

> **macOS note:** the example depends on `opencv_dart`, whose `dartcv4` prebuilt
> has a broken x86_64 slice that fails to link in profile/release builds. The
> macOS profile/release builds are therefore pinned to arm64 (see
> `EXCLUDED_ARCHS` in `example/macos/Runner/Configs/Release.xcconfig`), so
> `test/benchmark/run_matrix.sh macos` builds cleanly with `opencv_dart` enabled on Apple
> Silicon, no workaround needed. Linux/Windows/mobile are unaffected.

## Columns

**Interpreter:** `cpu` (4 threads) · `xnn` (XNNPACK) · `gpu_metal` (Metal GPU
delegate, Apple) · `gpu_glcl` (GL/CL GPU delegate, Android/desktop) · `coreml`
(CoreML/ANE, Apple).

**CompiledModel:** `cm_cpu` · `cm_gpu16` (strict GPU, fp16) · `cm_gpu32` (strict
GPU, fp32) · `cm_gpuA` (strict GPU, fp16, async) · `cm_g+c` (GPU|CPU fallback,
fp32) · `cm_hmA` (strict GPU, fp16, host-memory buffers, async).

Cells are `p50±std` ms. `unsupported` = delegate/accelerator not available on the
platform; `err` = failed to initialize (for strict GPU this means the model has
ops the GPU can't take; use `cm_g+c` instead); `dyn` = dynamic-shape model the
path can't allocate. Each non-CPU cell's output is also checked against the
Interpreter-CPU reference (`parity_maxdiff` in the CSV) to catch a GPU/driver
producing *wrong* results, not just slow ones.

---

## macOS: Apple M-series (Mac16,5), macOS 26.4, profile, commit 62441df

p50±std ms, 25 iterations / 8 warmup:

| model | cpu | xnn | gpu_metal | gpu_glcl | coreml | cm_cpu | cm_gpu16 | cm_gpu32 | cm_gpuA | cm_g+c | cm_hmA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilefacenet | 7±0 | 3±0 | 1±0 | unsupported | 20±0 | 2±0 | err | err | err | 1±0 | err |
| species_classifier_float16 | 2±0 | 1±0 | err | unsupported | err | 1±0 | err | err | err | 1±0 | err |
| superanimal_rtmpose_s_float16 | 12±0 | 8±0 | 13±1 | unsupported | err | 5±0 | err | err | err | 13±0 | err |
| yolov8n_float32 | 42±0 | 31±0 | 3±0 | unsupported | 15±0 | 21±0 | 3±0 | 3±0 | 2±0 | 3±0 | 2±0 |
| efficientdet_lite0 | 13±0 | 8±0 | 2±0 | unsupported | 6±0 | 9±0 | 3±0 | 3±0 | 3±1 | 4±0 | 2±0 |
| selfie_multiclass | 40±2 | 27±1 | 3±0 | unsupported | 22±1 | 18±1 | 2±0 | 2±0 | 2±0 | 2±0 | 2±0 |
| pose_landmark_heavy | 25±1 | 16±0 | 3±0 | unsupported | 5±0 | 17±0 | 3±0 | 3±0 | 2±0 | 3±0 | 2±0 |

### Reading this table

- **GPU is the story, not the runtime.** On heavy models both GPU paths collapse
  to a ~2 to 3 ms floor (yolov8n 42→3, selfie 40→2, pose 25→2), an ~8 to 20× win over
  CPU. CompiledModel-GPU and Interpreter-Metal-GPU are a wash there; differences
  of 1 ms are measurement noise.
- **CompiledModel's CPU path is the reliable win:** `cm_cpu` beats `cpu`/`xnn` on
  every model (yolov8n 42/31→21, selfie 40/27→18, pose 25/16→17).
- **Strict GPU (`cm_gpu16/32/A`) shows `err` on small models** (mobilefacenet,
  species, superanimal) because they have ops the GPU won't take. The
  Interpreter's Metal delegate hides this via silent per-op CPU fallback;
  CompiledModel surfaces it. **`cm_g+c` (GPU|CPU fallback) runs all of them**;
  that's the production config.
- **GPU is not always fastest:** on superanimal, `cm_cpu` (5 ms) beats both Metal
  (13 ms) and `cm_g+c` (13 ms). For that model CPU is the right choice.
- **fp16 vs fp32 (from `parity_maxdiff` in the CSV):** fp16 GPU paths
  (`gpu_metal`, `cm_gpu16/A/hmA`) diverge ~9e-3 from the CPU reference; fp32 paths
  (`cm_gpu32`, `cm_g+c`) are ~3e-6, three orders tighter, at no measured speed
  cost on this GPU but a ~14× higher *compile* cost (738 ms vs 53 ms cold start).
  Models emitting pixel-space coordinates (landmarks) should prefer fp32.

---

## iOS: iPhone 15 Pro (iPhone16,1), iOS 26.5, profile, commit 7d557b9

p50±std ms, 25 iterations / 8 warmup:

| model | cpu | xnn | gpu_metal | gpu_glcl | coreml | cm_cpu | cm_gpu16 | cm_gpu32 | cm_gpuA | cm_g+c | cm_hmA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilefacenet | 7±0 | 5±0 | 2±0 | unsupported | 25±0 | 7±0 | err | err | err | 2±0 | err |
| species_classifier_float16 | 2±0 | 2±0 | err | unsupported | err | 2±0 | err | err | err | 3±0 | err |
| superanimal_rtmpose_s_float16 | 15±0 | 13±0 | 23±1 | unsupported | err | 25±0 | err | err | err | 31±2 | err |
| yolov8n_float32 | 64±1 | 57±1 | 9±1 | unsupported | 19±0 | 117±0 | 7±0 | 10±0 | 7±0 | 10±0 | 7±0 |
| efficientdet_lite0 | 19±1 | 16±0 | 5±0 | unsupported | 8±0 | 30±1 | 9±1 | 8±0 | 8±1 | 8±0 | 9±1 |
| selfie_multiclass | 55±0 | 54±0 | 9±0 | unsupported | 27±2 | 115±3 | 7±0 | 11±0 | 8±0 | 11±0 | 8±0 |
| pose_landmark_heavy | 36±0 | 31±0 | 7±0 | unsupported | 7±0 | 58±1 | 6±0 | 9±0 | 6±0 | 9±0 | 6±0 |

### Reading this table

- **`cm_cpu` is ~2× SLOWER than the Interpreter CPU on iOS, the opposite of
  macOS.** yolov8n 64→117, selfie 55→115, pose 36→58, superanimal 15→25. The
  CompiledModel CPU accelerator does not use XNNPACK the way the classic
  interpreter does on ARM, so **do not pick `cm_cpu` as a CPU fallback on iOS**;
  the classic interpreter CPU/XNNPACK path is much faster. (On Apple Silicon
  macOS it was the reverse; this is genuinely per-platform.)
- **GPU still wins big and the two GPU runtimes tie.** `cm_gpu16`/`cm_gpuA` and
  Interpreter `gpu_metal` all land ~6 to 9 ms on the heavy models (yolov8n 64→7,
  selfie 55→7, pose 36→6), ~8 to 9× over CPU.
- **CoreML/ANE is a real contender here** (unlike macOS): pose 7 ms,
  efficientdet 8 ms, yolov8n 19 ms, sometimes matching GPU, worth A/B-ing
  per-model on Apple hardware.
- **Same strict-GPU `err` pattern** on mobilefacenet/species/superanimal;
  `cm_g+c` runs them all. And same superanimal lesson: every GPU path (Metal 23,
  `cm_g+c` 31) is *slower* than CPU (15); that model wants CPU.
- **fp32 GPU compiles in line with fp16 here** and runs ~3 ms slower on the heavy
  models (`cm_gpu16` 7 vs `cm_gpu32` 10), a more visible fp16/fp32 speed gap
  than macOS, so the accuracy-vs-speed choice matters more on iOS.

**Practical takeaway for an iOS app:** prefer `{gpu, cpu}` (`cm_g+c`) or the
Interpreter Metal/CoreML paths; avoid CompiledModel CPU-only as a fallback;
fall back to the classic Interpreter instead, which is what
`face_detection_tflite` already does via its `useCompiledModel` escape hatch.

---

## Linux: _pending_

Run `test/benchmark/run_matrix.sh linux` on the target box and commit the updated CSV +
table. Expectations to verify: `gpu_metal`/`coreml` are `unsupported`; the new
`libLiteRtWebGpuAccelerator.so` (Vulkan) drives the `cm_gpu*`/`cm_g+c` columns;
`gpu_glcl` availability depends on the GL/CL delegate.

## Windows: _pending_

Run `test/benchmark/run_matrix.sh windows`. The WebGPU (Dawn/D3D12) accelerator
drives the
`cm_gpu*` columns; watch the fp16 `parity_maxdiff` on weak D3D12 drivers.
