# CompiledModel (LiteRT Next) spike findings

Archived 2026-06-16 from the now-removed `spike/compiled_model/` proof-of-concept.
The CompiledModel API shipped in release 3.0.0 (commit `271e7f2`); this file
preserves the empirical findings, API idioms, and upstream-bug repros the spike
established so the detail is not lost when the scratch code goes away. Internal
reference (not consumer-facing); it documents internal profiling data and two
unfiled upstream bugs.

All numbers were measured on macOS arm64 unless noted, LiteRT `ai-edge-litert`
2.1.5, warm loops, median µs unless noted. The Interpreter side is the bundled
classic `libtensorflowlite_c-mac` (TF 2.20.0), so Interpreter-vs-CompiledModel
multiples mix "GPU" plus "newer runtime"; the GPU-vs-CompiledModel-CPU
comparison is the clean one.

## Headline

- Median CompiledModel GPU-async vs Interpreter-XNNPACK: ~19× (32-model sweep).
- Median CompiledModel-CPU vs Interpreter-XNNPACK: ~2×.
- CompiledModel beats the classic Interpreter everywhere (CPU and GPU).

Selected (median µs, `sweep.dart`):

| model | size | interp_xnn | cm_cpu | gpu_sync | gpu_async | best |
|---|--|--:|--:|--:|--:|--:|
| selfie_multiclass (seg) | 16M | 98,425 | 17,058 | 1,606 | 1,349 | 73× |
| yolov8n | 12M | 107,199 | 20,129 | 1,700 | 1,488 | 72× |
| efficientdet_lite2 | 22M | 83,609 | 25,109 | 2,586 | 2,391 | 35× |
| pose_landmark_heavy | 26M | 49,506 | 16,551 | 2,091 | 1,866 | 27× |
| face_detection_back | 308K | 5,588 | 3,786 | 768 | 285 | 20× |
| face_detection_short_range | 224K | 927 | 575 | 784 | 187 | 5× |

Rule: GPU wins big above ~1 to 2 ms of compute; tiny sub-ms models favor CPU.
It is compute, not category: `face_detection_back` (heavy) wins on GPU,
`face_detection_front` (tiny) on CPU.

## Sync vs async

Synchronous per-call timing pays full encode to submit to sync every call and
makes the GPU look bad on small models. Async pipelining
(`LiteRtRunCompiledModelAsync`) gives the GPU 2.8 to 4.9× throughput; the CPU
gains ~nothing. This flips the small-model verdict:

| face_detection_short_range | sync | async-8 |
|---|--:|--:|
| GPU/Metal | 773 µs | 158 µs |
| CPU | 566 µs | 580 µs |

So with an async-pipelined design (a live camera feed) GPU wins across the
board, including tiny face models. Caveat: async numbers are throughput, not
single-shot latency, and the benchmark feeds a static input (no per-frame upload
cost). Single-shot `runAsync` (event-poll) is ~2× faster than sync on GPU and
~matches a depth-8 pipeline; CPU runs synchronously (asyncOut=0). Single-shot
async captures the win; no streaming API needed.

### Async recycle idiom

`LiteRtGetTensorBufferEvent` is borrowed (do NOT destroy) then `LiteRtWaitEvent`
then `LiteRtClearTensorBufferEvent(buf)` before reusing the output slot.
(`LiteRtSetTensorBufferEvent(buf, null)` returns InvalidArgument; wrong
approach.)

### Blocking wait replaces event polling (2026-06-10)

Verified against official LiteRT source (commit ea79caf): `LiteRtTensorBufferT::Lock`
waits on an attached event with timeout -1 (`tensor_buffer.cc:988`), sync Run
waits the same way (`compiled_model.cc:1817 to 1833`), and no official code polls
with timeout 0. `runAsync` therefore does a blocking `LiteRtWaitEvent(event, -1)`
per output instead of `waitEvent(0)` plus `Future.delayed(Duration.zero)` poll
loop (which also had an unbounded-loop edge case). In-app (debug,
`cm_inference_only_test`, median ms, GPU|CPU runAsync): parity within noise. Sync
`run()`'s extra ~0.9 ms on GPU is NOT the event wait (it persists either way);
async dispatch plus event wait stays ~3× faster than sync `run()` on Metal even
when the wait blocks.

## FP16 vs FP32

GPU defaults to FP16 (fast); FP32 via opaque `gpu_options` TOML `precision=2`.
FP16 faster as expected (yolov8n FP16 1676 vs FP32 2058 µs). Ship FP16 default,
FP32 opt-in (GPU-only; CPU is always FP32).

## Accelerator fallback (GPU|CPU)

Google's `HardwareAccelerator` docstring: "Using GPU or NPU alone may fail. For
robust execution, combine with CPU as fallback: `GPU | CPU`." Confirmed: GPU-only
(mask 2) returns `504 (Compilation)` for several models; GPU|CPU (mask 3) makes
them compile/run:

| model | CPU | GPU(2) | GPU\|CPU(3) |
|---|--:|--:|--:|
| mobilefacenet | 2309 | 504 | 626 (3.7× faster) |
| species_classifier_f16 | 1152 | 504 | 1434 |
| superanimal_ssdlite_f16 | 4972 | 504 | 5895 |
| superanimal_rtmpose_f16 | 5363 | 504 | 11350 |

Fallback makes them run, not always run fast (partition overhead can exceed the
GPU gain). The failures are NOT dtype: every failing model has Float32 I/O, no
quantization (cross-checked via the classic TFLite C API). The `504`s are
unsupported GPU ops (`DEQUANTIZE`, `RELU_0_TO_1`, `L2_NORMALIZATION`), fixed by
fallback. Status `3 = RuntimeFailure`, `504 = Compilation` (confirmed via
`LiteRtGetStatusString`).

## Two confirmed upstream bugs (not yet filed)

Reproduced with Google's own Python API (`CompiledModel.from_file(model,
hardware_accel=GPU|CPU)`), so NOT our FFI:

- `face_detection_full_range_sparse` to SIGABRT: `DENSIFY: Operation is not
  supported` to uncaught `std::bad_optional_access`. Crashes even with CPU
  fallback. Classic Metal runs it fine.
- `gesture_embedder` to `RuntimeError` at `litert_compiled_model.h:1486`: GATHER
  shape mismatch / Metal kernel `Unable to parse bc coord for BATCH axis`. Fails
  even with fallback. Classic Metal runs it fine.

Repros were staged for reporting but never filed.

## Host-memory I/O: official, working, opt-in

Official host-side zero-copy uses `LiteRtCreateTensorBufferFromHostMemory`: the
caller owns a `LITERT_HOST_MEMORY_BUFFER_ALIGNMENT`-aligned buffer for the whole
tensor-buffer lifetime, and LiteRT wraps it at buffer creation. This is distinct
from holding a transient lock pointer open while preprocessing/decoding; that
lock-callback experiment was removed because it crashed under concurrent
multi-model Metal load. Outputs were bit-identical to managed buffers in all
successful modes, but performance was model/accelerator-dependent and often
slower on Metal GPU.

Decision: the package API exposes `TensorBufferMode.hostMemory` and the direct
`writeInput`/`dispatchAsync`/`readOutput` API as opt-in only;
`TensorBufferMode.managed` remains the default. Host memory is correct and
sometimes shaves ~1 to 2% on CPU/fallback-heavy models, but is not a safe
default and is often slower on Metal GPU. Managed buffers remain the default for
GPU-heavy face/segmentation models. The animal models are fallback-heavy on
Metal (`DEQUANTIZE` / `RELU_0_TO_1` unsupported), so host memory does not change
the main bottleneck. For true GPU zero-copy, the next frontier is GPU-native
buffer interop (camera textures / platform GPU buffers), not host-memory
wrapping from Dart arrays.

## Isolates

Metal works inside a worker isolate. Cross-isolate overhead: plain
`SendPort.send` ~+230 µs on face_detection_back, `TransferableTypedData` ~+20 µs
(near-free). Isolate value is UI-thread offload, not inference speed.

## In-app per-model engine table (`cm_inference_only_test`, median ms)

| model | interp | cm CPU run | cm GPU\|CPU run | cm GPU\|CPU runAsync |
|---|--:|--:|--:|--:|
| face_detection_back | 6.16 | 4.11 | 1.36 | 0.44 |
| face_landmark | 1.39 | 0.84 | 0.57 | 0.49 |
| iris_landmark | 1.47 | 0.50 | 1.09 | 0.65 |

Detection plus mesh belong on GPU|CPU async; iris (64×64) is fastest on
CompiledModel-CPU, consistent with the "GPU wins above ~1 to 2 ms compute" rule.
face_detection_tflite pins iris to `{cpu}`.

## Whole-pipeline A/B variance lesson (2026-06-10)

The same `compiledmodel_ab_test` on the same code measured CM fast-mode at 0.85×
(midday, loaded machine) and 1.31× (quiet machine). The interpreter engine was
unchanged between runs; absolute numbers and cross-run ratios from this
benchmark are unreliable. Only the within-run paired ratio on a quiet machine is
meaningful. Post-change quiet-machine runs reproduce within ±0.03: fast
1.31×/1.32×, full 1.33×/1.30× (1 face); 1.03×/1.04× (4 faces,
decode-dominated).

## iOS: packaging, accelerator discovery, API-version pin (2026-06-11)

Verified against the LiteRT source:

- Accelerator discovery is filename-based with no usable process fallback for
  the Metal plugin. `RegisterGpuAccelerator` dlopen's
  `<RuntimeLibraryDir>/libLiteRtMetalAccelerator.dylib` and looks up the exported
  `LiteRtAcceleratorImpl` def. The RTLD_DEFAULT fallback only applies to a
  `LiteRtRegisterGpuAccelerator` function symbol, which the official Metal
  accelerator does NOT export. The accelerator file must keep its exact dylib
  name on disk; pre-dlopen'ing it ourselves cannot replace the directory scan.
- iOS packaging uses library-type xcframeworks (`xcodebuild -create-xcframework
  -library libLiteRt.dylib ...`), NOT `.framework` bundles. Frameworks force a
  binary rename (no `lib` prefix / `.dylib` suffix), which breaks the scan;
  bare-dylib xcframework slices are embedded under their original names. The Dart
  loader opens `<app>/Frameworks/libLiteRt.dylib` and passes that directory as
  `kLiteRtEnvOptionTagRuntimeLibraryDir`.
- iOS prebuilts must be pinned to the v2.1.5-era C API. Commit `1ac2a58f`
  (2026-06-01) added a leading `LiteRtEnvironment` parameter to
  `LiteRtCreateModelFromFile/FromBuffer`; later prebuilts crash v2.1.5-shaped
  bindings. Pin: commit `1adc2475829fbe52d5670873821a45bea8779532` (2026-05-28).
- iOS-simulator Metal event wedge (sequence-dependent): closing a Metal model
  that ran `runAsync` and then waiting on an async event from a freshly created
  environment can hang forever in `MTLSimSharedEvent waitUntilSignaledValue`.
  Simulator-only driver path (MTLSimDriver); real devices use a different Metal
  stack. The integration test keeps the fallback test sync-only so CI stays
  deterministic.

## Android: CompiledModel CPU path (2026-06-11)

- Bundled `libLiteRt.so` from the API-pinned prebuilt commit (`1adc2475`,
  arm64-v8a plus x86_64, both 16 KB page-size aligned) via a Gradle
  download-at-build task, additively next to the classic Maven artifacts. All 38
  bound symbols are exported (with `@@VERS_1.0` symbol versioning; plain-name
  dlsym still resolves them).
- The OpenCL/GL GPU accelerator is deliberately NOT bundled yet: when
  `libLiteRtClGlAccelerator.so` registers in an environment without working
  OpenCL (every emulator), `LiteRtCreateCompiledModel` with GPU|CPU fails with
  status 3 (RuntimeFailure) instead of falling back to CPU. Re-add once validated
  on real hardware.

## C API call sequence (from official `litert_cc_sdk` headers)

```
LiteRtCreateEnvironment(0, NULL, &env)
LiteRtCreateOptions(&opts); LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu /*=1*/ | Gpu /*=2*/)
LiteRtCreateModelFromFile("model.tflite", &model)
LiteRtCreateCompiledModel(env, model, opts, &compiled)
// tensor I/O:
//   LiteRtGetCompiledModelInputBufferRequirements + LiteRtCreateManagedTensorBufferFromRequirements
//   LiteRtLockTensorBuffer / write / LiteRtUnlockTensorBuffer
//   LiteRtRunCompiledModel(compiled, sigIndex, nIn, inBufs, nOut, outBufs)
LiteRtDestroyCompiledModel / Model / Options / Environment
```
