# Graph shape vs delegate: XNNPACK's TRANSPOSE_CONV is the slow path

Status: **five findings, two shipped patches, and one conclusion that reversed on
contact with a real phone.**

Read finding 5 first if you are short of time. Findings 1 to 4 are all macOS on an
M4 Max, and on an iPhone 15 Pro the GPU delegate turns out to be worth about 1% rather
than the 5x the Mac suggested. What survives as actionable is smaller and duller than
the headline: **two delegates silently no-op on the shipped models, so iOS pays ~20% for
nothing**, and both patches here fix correctness rather than buy speed.

Two patches came out of this and both are verified: `patches/gpu_transpose_conv_v4.patch`
(finding 4, vendored into the macOS GPU dylib) and `patches/coreml_mean_padding.patch`
(finding 3, verified on macOS and on device). A third claim, a CoreML stall needing a
partition-bounding fix, was made and then **retracted** as a harness artifact; it is
documented below so the trap is not re-entered.

Follows on from [`macos_transpose_conv_gap.md`](macos_transpose_conv_gap.md) (ruy
multithreading fix) and
[`delegate_verification.md`](delegate_verification.md) (silent delegate no-ops and
the `CompiledModel` corruption). Neither is contradicted here. This document adds
what happens once you have **two variants of the same model that differ only in
graph shape**.

## The two variants

The dog and cat landmark models (`small_v3large_384_long`, MobileNetV3Large + a
4-deconv heatmap head) were re-exported from identical weights via a batch-1
concrete function instead of `from_keras_model`. That constant-folds away the
`PACK`-derived `TRANSPOSE_CONV` output shapes:

| | dynamic (shipped) | static |
|---|---|---|
| ops | 295 | 279 |
| `SHAPE` / `STRIDED_SLICE` / `PACK` | 5 / 5 / 7 | 0 / 0 / 1 |
| predictions | reference | identical to 6.6e-7 |

Both files, and the cat equivalents, are listed at the bottom.

## Finding 1: XNNPACK's TRANSPOSE_CONV is ~1.8x slower than the built-in kernel

Dog landmarks_384, macOS arm64 (M4 Max), flutter_litert 3.7.0, median `invoke()`.
`none` means no delegate at all, i.e. `PerformanceMode.disabled`.

| graph | delegate | t=1 | t=4 | thread scaling | val NME_IOD |
|---|---|---|---|---|---|
| dynamic | none | 97.14 ms | 33.24 ms | 2.92x | 8.5664 |
| dynamic | **xnnpack** | 92.68 ms | **28.14 ms** | 3.29x | 8.5664 |
| static | none | 97.74 ms | 33.90 ms | 2.88x | 8.5664 |
| static | **xnnpack** | 230.71 ms | **60.65 ms** | 3.80x | 8.5664 |

Accuracy is identical in all four cells (full 480-image DogFLW val split, on the
converted files), so nothing here is a correctness question.

Read it in three steps:

1. **With no delegate the two graphs are indistinguishable** (33.24 vs 33.90 ms).
   Same weights, same ops. Graph shape is irrelevant to the built-in kernels, which
   rules out the shape itself being the cause of anything downstream.
2. **static + xnnpack is 1.79x slower than static + nothing** (60.65 vs 33.90), and
   2.36x slower at one thread (230.71 vs 97.74). The delegate is actively harmful
   on this graph. The only thing it can newly claim, relative to the dynamic graph,
   is the deconv head.
3. **dynamic + xnnpack is the fastest cell** (28.14 ms) but beats no-delegate by
   only 1.18x.

So the conclusion is inverted from the obvious one:

> The dynamic graph is not fast because dynamic shapes are good. It is fast because
> its dynamic tensors make XNNPACK **decline the deconv region**, leaving it on the
> built-in ruy kernel restored by the fix documented in
> [`macos_transpose_conv_gap.md`](macos_transpose_conv_gap.md). XNNPACK then
> contributes only the backbone.

### Consequences

- **`PerformanceConfig`'s doc comment oversells XNNPACK for this model class.** It
  says "XNNPACK (2-5x SIMD acceleration)"; measured here it is **1.18x**
  (33.24 -> 28.14 ms). Worth softening, or qualifying by op mix.
- **There is probably unclaimed upside.** If XNNPACK took the backbone and the 1x1
  convs while `TRANSPOSE_CONV` stayed on ruy, a static graph should beat 28.14 ms.
  The dynamic graph gets that split by accident, and accident is unlikely to be the
  optimal partition: the dynamic tensors probably push XNNPACK off more than just
  the deconvs. An op-exclusion knob (or an internal exclusion for
  `TRANSPOSE_CONV` on this platform) would let the split be chosen deliberately.
  **Untested**, because there is no such flag to test with today.
- The 3.80x thread scaling in the static+xnnpack row is the best scaling in the
  table and still the slowest cell. Scaling ratios are not a proxy for speed here.

## Finding 2: Metal cannot take the deconv head, for a different reason

`delegate_verification.md` records `dev = 0.0` (attached, delegated nothing) for
GPU and CoreML on these models, and root-causes the `CompiledModel` corruption to a
dynamic model output tensor. The static variant removes that tensor, so it is the
natural test.

Engagement detected by the `dev != 0.0` method from that document: run CPU-only as
reference, then under each delegate, and compare output.

| graph | backend | outcome | dev |
|---|---|---|---|
| dynamic | xnnpack | engaged | 2.170e-05 |
| dynamic | **metal** | **create-failed**, then `mutex.cc RAW: Lock blocking` | -- |
| dynamic | coreml | delegate never initialises (see finding 3) | -- |
| static | xnnpack | engaged | 2.176e-05 |
| static | **metal** | **engaged** | **5.364e-07** |
| static | coreml | delegate never initialises (see finding 3) | -- |

Both Metal rows were measured on the same machine within the same minute, under the
same GPU load, so the difference is the graph and not the environment.

**But the win is not what it looks like.** With the static graph the Metal delegate
prints:

```
ERROR: Following operations are not supported by GPU delegate:
DEQUANTIZE:
TRANSPOSE_CONV: Max version supported: 3. Requested version 4.
123 operations will run on the GPU, and the remaining 156 operations will run on the CPU.
```

and lands at **28.80 ms** (p10 28.00, p90 30.85) against 28.14 ms for
dynamic + XNNPACK. A wash.

So there are **two independent blockers stacked**, and the doc's dynamic-output-tensor
explanation is only the outer one:

1. Dynamic model output tensor: breaks interpreter creation entirely.
2. **`TRANSPOSE_CONV` version 4 requested, GPU delegate supports max version 3.**
   Fixing (1) exposes (2). The expensive head stays on CPU either way.

`DEQUANTIZE` being unsupported compounds it: fp16 weights generate 140 of those, so
the graph fragments into GPU and CPU regions regardless.

### Consequences

- **Switching iOS to a static export would buy nothing.** It converts a hard
  failure into a partial delegation at the same latency. Good news for the
  dependent packages: the current dynamic export is not leaving an iOS win unclaimed.
- **The `TRANSPOSE_CONV` v3-vs-v4 gap turned out to be the real lead, and is now fixed.**
  It was not a stale-dylib problem: v4 is precisely "this op carries a fused activation",
  which Keras always produces by folding `Conv2DTranspose -> BatchNorm -> ReLU`. See
  finding 4 for the patch and finding 5 for why it matters less than it appears.
- A float32 export would isolate how much the `DEQUANTIZE` fragmentation costs, at
  2x the file size. Untested.

## Finding 3: CoreML cannot compile these models (MEAN padding type)

Scope note first, because an earlier draft of this document got it wrong: there is
**no** flutter_litert-side stall here, and no partition-bounding fix is needed. That
claim was a harness artifact and is retracted. See the method note on TensorFlow
in-process below.

### The real failure

The TFLite-to-CoreML conversion emits a pooling layer for `MEAN` without setting a
padding type, and CoreML's validator rejects it:

```
Error compiling model compiler error: Error reading protobuf spec. validator error:
Padding type for the pooling layer 'PoolingLayerBuilder (MEAN)_32' is not set.
ERROR: Failed to Compile and save Model.
ERROR: CoreMl Kernel was not initialized
ERROR: Node number 279 (TfLiteCoreMlDelegate) failed to prepare.
ERROR: Restored original execution plan after delegate application failure.
```

Reproduced on **three** models: `species_classifier_float16`, and both the dynamic
and static landmark graphs. All contain `MEAN` ops (8 in the landmark models, from
MobileNetV3 squeeze-excite blocks and global pooling), so this is not specific to one
architecture.

Behaviour is otherwise clean: it fails in **0.1 s**, TFLite restores the original
execution plan, and the model runs correctly on CPU. That is exactly why
`delegate_verification.md` measured `dev = 0.0` for CoreML, and the two observations
agree. `max_delegated_partitions` makes no difference: bounded and unbounded both
fail in well under a second.

**Fixed and verified.** `patches/coreml_mean_padding.patch` sets `mutable_valid()` before
that early return, applied by `build-coreml-macos.yml` and `build-coreml-ios.yml`. On
macOS the delegate goes from "interpreter creation failed" to engaged at 27.51 ms with
NME_IOD 8.5660 against a CPU reference's 8.5664. On an iPhone 15 Pro it goes from
deviation exactly 0.0, meaning attached and delegating nothing, to 9.5e-04, which is fp16
rounding rather than corruption.

It buys no speed: 47.89 ms on device against XNNPACK's 47.82 ms. What it removes is a
**silent regression**, since before the fix selecting `PerformanceMode.coreml` cost
57.92 ms because failing to compile drops the model to bare CPU without XNNPACK.

7 of the 8 models shipped across dog_detection, cat_detection and animal_detection contain
a `MEAN` op and were therefore affected. Only `superanimal_hrnet_w32` has none, which makes
it a misleading model to smoke-test a Core ML delegate against. The omission itself is
upstream in TFLite's layer builder and is worth reporting there.

One cosmetic observation: `CoreMlDelegateOptions` defaults `coremlVersion` to 0,
which is not a valid value, so every creation logs
`coreml_version must be 2 or 3. Setting to 3.` Harmless noise on each init;
defaulting to 3 explicitly would silence it.

## Finding 4: lifting the TRANSPOSE_CONV v4 gate is worth 5.25x, and the patch is two lines

This is the actionable one. Finding 2 established that Metal refuses the deconv head
with `TRANSPOSE_CONV: Max version supported: 3. Requested version 4.` and concluded
the head stays on CPU either way. That conclusion was too pessimistic: it is worth
finding out *why* v4 is requested.

### Why the models declare v4

The converter folds `Conv2DTranspose -> BatchNorm -> ReLU` into one op. The result has
four inputs (a folded-BN bias) and, critically,
`TransposeConvOptions.fusedActivationFunction = 1` (RELU). Carrying a fused activation
is what pushes the opcode to version 4. So the models genuinely use the v4 feature;
this is not a gratuitous version bump, and simply relabelling the opcode would silently
drop the ReLU.

### The gate is the only blocker

Proven two ways. First a deliberately-wrong probe: downgrade the opcode to v3 and clear
the activation. Metal then accepted the **entire graph** with no unsupported-op
messages at all, ran it at **5.11 ms**, and computed it correctly (GPU versus its own
CPU, dev 1.06e-05). Output differed from the original by 4.86e-01, confirming the
ReLUs really were removed rather than relocated.

Then the legitimate version: move each ReLU out into a separate `RELU` op, which the
delegate already supports, keeping the original output tensor identity so nothing
downstream is rewired.

    TRANSPOSE_CONV(act=RELU) -> T     becomes
    TRANSPOSE_CONV(act=NONE) -> T_pre  then  RELU(T_pre) -> T

Measured over all 480 DogFLW val images, macOS arm64 (M4 Max), flutter_litert 3.7.0:

| model | backend | NME_IOD | median `invoke()` |
|---|---|---|---|
| shipped dynamic | xnnpack | 8.5664 | 26.83 ms |
| static, v4 | metal | 8.5664 | 30.25 ms |
| **static, ReLU unfused** | **metal** | **8.5665** | **5.11 ms** |
| static, ReLU unfused | xnnpack | 8.5664 | 58.00 ms |

Unfused versus shipped on CPU: max per-coordinate difference **6.6e-07**, i.e.
numerically identical. Unfused on Metal versus on XNNPACK: **1.7e-04**, consistent with
GPU fp16 precision. **5.25x** against the best currently shipped configuration, at
unchanged accuracy and unchanged file size.

### The patch

Both helpers are already linked into the shipped `libtensorflowlite_gpu-mac.dylib`:

```
tflite::gpu::MaybeFuseActivation(TfLiteFusedActivation, GraphFloat32*, Node*)
tflite::gpu::CheckMaxSupportedOpVersion(TfLiteRegistration const*, int)
tflite::gpu::TransposeConvBuiltinOperationParser::IsSupported(...)
tflite::gpu::TransposeConvBuiltinOperationParser::Parse(...)
```

`MaybeFuseActivation` is generic over `TfLiteFusedActivation` and other parsers such as
`CONV_2D` already call it. In
`tensorflow/lite/delegates/gpu/common/model_builder.cc`:

1. `TransposeConvBuiltinOperationParser::IsSupported`: raise
   `CheckMaxSupportedOpVersion(registration, 3)` to `4`.
2. `TransposeConvBuiltinOperationParser::Parse`: add the
   `MaybeFuseActivation(tf_options->activation, graph, node)` call the conv parser
   already makes.

Those call sites are inferred from the binary's symbol table, not from reading the
source, so confirm against the actual file. But the shape of the change is clear and
uses machinery that is already present.

**The patch is de-risked in advance.** The unfused model proves the GPU kernels handle
this graph correctly at 5.11 ms, so the only open question is parser plumbing, not
kernel support.

The real cost is not the two lines, it is rebuilding the GPU dylib for macOS, iOS and
Android. The x86_64 cross-compile blocker recorded in
[`macos_transpose_conv_gap.md`](macos_transpose_conv_gap.md) sits in that path.

### Why fixing it here beats fixing it per-model

`Conv2DTranspose -> BatchNorm -> ReLU` is the standard Keras pattern, so every
segmentation, pose and heatmap model exported this way hits the same wall. Patching the
delegate fixes all of them. The model-side rewrite fixes one file at a time, has to be
re-run on every export or the model silently loses GPU eligibility with no error, and
means hand-editing a serialized flatbuffer.

The workaround's one advantage is that it needs no rebuild and works against the
already-published 3.7.0, so it is a reasonable stopgap. It lives at
`dogs-in-the-wild-ml/scripts/unfuse_transpose_conv_relu.py` and handles RELU and RELU6.

### Caveats before anyone quotes the 5.25x

- **Measured on an M4 Max GPU.** Mobile GPUs are far weaker and the ratio will differ.
  `delegate_verification.md` has the landmark model at 32.46 ms on iOS GPU, but that was
  a no-op falling back to CPU, so iOS has never actually run this graph on GPU.
- **iOS may not need any of this.** `metal_delegate_native.dart` returns
  `tfliteBinding` on iOS and only opens the separate `libtensorflowlite_gpu-mac.dylib`
  on macOS, so the iOS Metal path is a different binary and may already accept v4.
  Worth a simulator compatibility check, which is valid for op support even though
  simulator *timings* are host-GPU and meaningless for device performance.
- **Medians exclude the first 10 invocations.** GPU delegates carry warmup and
  host-to-device transfer cost. For single-shot photo inference rather than a video
  stream the amortised figure will be worse.
- **A model built for the GPU path is slower on XNNPACK.** The unfused static graph is
  58.00 ms under XNNPACK against the shipped model's 26.83, for the reason in finding 1.
  So the model and the delegate choice have to move together; swapping the asset alone
  would be a 2x regression wherever auto-mode picks XNNPACK, which is macOS and Android.

## Finding 5: on device, all three compute units tie, and two of them silently no-op

Everything above is macOS on an M4 Max: 40 GPU cores. An iPhone 15 Pro has 6. Since
`InterpreterFactory` auto-mode picks XNNPACK on macOS and the GPU delegate only on iOS, the
Mac is the platform that matters least and was the only one measured.

Measured on a physical iPhone 15 Pro, dog landmark model, median of 30 invocations after 5
warmups, deviation against a no-delegate reference on the same fixed input:

| variant | backend | median | dev vs CPU | verdict |
|---|---|---|---|---|
| dynamic (ships today) | none | 54.98 ms | -- | reference |
| **dynamic (ships today)** | **xnnpack** | **47.82 ms** | 7.0e-06 | engaged |
| dynamic (ships today) | gpu | 57.22 ms | **0.0** | NO-OP, bare CPU |
| dynamic (ships today) | coreml | 57.92 ms | **0.0** | NO-OP, bare CPU |
| static, v4 fused ReLU | gpu | 53.14 ms | 7.8e-07 | engaged |
| static, v4 fused ReLU | coreml (patched) | 47.89 ms | 9.5e-04 | engaged |
| static, ReLU unfused | gpu | **46.65 ms** | 3.5e-05 | engaged |
| static, ReLU unfused | coreml (patched) | 56.29 ms | 8.9e-04 | engaged |

Four things follow.

1. **The GPU is worth ~1% on a phone.** Best CPU 47.82 ms, best GPU 46.65 ms. The same
   graph does 5.11 ms on an M4 Max, so GPU time scales roughly with core count while the
   CPU path does not.
2. **XNNPACK 47.82, GPU 46.65, ANE 47.89.** Three unrelated compute units within 2%
   strongly suggests this model is **memory-bandwidth bound** on an A17 Pro rather than
   compute bound.
3. **iOS already accepts `TRANSPOSE_CONV` v4.** `static_v4 + gpu` engages here where the
   stock macOS delegate refuses it, so finding 4's patch is macOS-only in value.
4. **Two silent no-ops are live in the shipped configuration.** Both the GPU and CoreML
   delegates attach and delegate zero ops on the model as shipped, so the stage runs on
   bare CPU *without* XNNPACK. Since auto-mode sends iOS to GPU, iOS is paying 57.22 ms
   instead of 47.82, about 20% for nothing, matching the "17-23% latency, silently" figure
   in [`delegate_verification.md`](delegate_verification.md).

### Consequences for this library

- **`_createAutoMode` sending iOS to the GPU delegate is a pessimisation for any model the
  delegate declines.** It is strictly worse than XNNPACK there, because declining falls
  back to bare CPU rather than to the next-best delegate. Worth considering whether
  auto-mode should verify non-zero delegation and fall back to XNNPACK when it gets none.
  `delegate_verification.md` already notes the detection mechanism is resolved for GPU.
- **`PerformanceConfig`'s "XNNPACK (2-5x SIMD acceleration)" doc comment measured 1.18x**
  on this model class.
- Per-stage overrides now exist downstream (`landmarkPerformanceConfig` in dog_detection
  and cat_detection, `posePerformanceConfig` in animal_detection) precisely because the
  right delegate differs per model. `hrnet_w32` is 4.8x faster on the GPU delegate with
  correct output, while `species_classifier` and both face localizers cannot create a GPU
  interpreter at all, so one pipeline-wide mode cannot express what these pipelines need.

## What this changes about the shipped guidance

Two dependent packages shipped a static re-export (`dog_detection` 2.0.1,
`cat_detection` 2.0.1) on the strength of flutter_litert **3.6.0** numbers, where
static was 1.56x faster, then reverted when 3.7.0 inverted it. Both are back on the
dynamic export.

The generalisable rule, which is now understood rather than merely observed:

> **Delegate latency is a property of the (model, runtime version, delegate) triple.**
> On 3.6.0 the built-in `TRANSPOSE_CONV` kernel was crippled, so pushing the deconv
> into XNNPACK was a win. The fix documented in
> [`macos_transpose_conv_gap.md`](macos_transpose_conv_gap.md) restored the built-in
> kernel, and the same change became a 2x loss.

For a deconv-headed model on 3.7.0, keep the dynamic export.

## Method notes

- Driven over ctypes against each release's own bundled
  `libtensorflowlite_c-mac.dylib`, reproducing `InterpreterFactory`'s macOS
  auto-mode (XNNPACK delegate, `numThreads = min(4, nproc)` = 4, `QS8|QU8`).
  `tf.lite` Python is a third runtime again and ranks these graphs differently from
  both dylibs; do not mix them.
- `TfLiteXNNPackDelegateOptionsDefault` writes **more than the 48 bytes** the Dart
  bindings declare. Handing it a 48-byte buffer corrupts the caller's stack and
  segfaults on return. Over-allocate. This dylib also defaults
  `weight_cache_file_descriptor` to -1, where `XNNPackDelegateOptions` in Dart
  leaves it 0 via `calloc`; worth confirming that is intentional.
- Timing cannot detect a no-op delegate. Use `dev != 0.0`. Conversely `dev != 0.0`
  does not mean *fully* delegated, as finding 2 shows: 123 of 279 ops was enough to
  register as engaged.
- **Do not import TensorFlow into a process that drives these dylibs.** This is
  what produced, and then retracted, the false "CoreML stalls for 10+ minutes"
  finding. TF ships its own copy of the TFLite and CoreML delegate symbols; with
  both loaded, CoreML delegate creation goes from **0.1 s to hanging past 120 s**.
  Measured A/B in one script, the only difference being an `import tensorflow`:

  | process | CoreML create |
  |---|---|
  | no tensorflow | 0.1 s, fails cleanly with the `MEAN` error |
  | tensorflow imported | hangs, killed at 120 s |

  XNNPACK and Metal tolerate it (their timings and `dev` values reproduce either
  way), so the corruption is CoreML-specific, but load the val data with plain numpy
  and keep TF out of the process regardless.
- Return the output of a **single invocation on a fixed input** when computing `dev`.
  An intermediate version of this work compared the last output of two timing loops
  with different iteration counts, which compared different images and reported
  `dev = 7.5e-01` for a delegate whose true `dev` is `5.4e-07`. That looks exactly
  like the upstream corruption signature and would have been reported as one.
- Re-verify anything load-bearing in a TF-free process. Findings 1 and 2 were both
  re-measured that way; finding 1's ratios were stable (1.79x vs 1.81x) even though
  absolute latencies rose ~30% under machine load, and finding 2's `dev` reproduced
  to the digit.
- Run each (model, backend) cell in its own process. Metal on the dynamic graph
  hangs on a mutex after failing creation and will take a single-process harness
  with it. `scripts/run_delegate_matrix.py` in the dogs repo does this.
- **CoreML options defaults are not zero, and one of them matters a lot.**
  `CoreMlDelegateOptions` in Dart defaults `minNodesPerPartition` to **2**, and
  `_createCoreml` overrides only `enabledDevices`. A hand-built options struct that
  zeroes the rest sets `min_nodes_per_partition = 0`, which lets CoreML create a
  partition per node. With 279 ops and both `DEQUANTIZE` and `TRANSPOSE_CONV`
  unsupported, that is a very large number of tiny partitions, each compiled
  separately on first use. That configuration hung for **over 15 minutes** and was
  a bug in the test harness, not in the library. Field order is
  `enabled_devices / coreml_version / max_delegated_partitions /
  min_nodes_per_partition`, four consecutive ints.

## Files

Identical weights, differing only in graph shape:

```
dog dynamic  ~/PycharmProjects/dogs-in-the-wild-ml/artifacts/small_v3large_384_long/dog_face_landmarks_384_float16.tflite
dog static   ~/PycharmProjects/dogs-in-the-wild-ml/artifacts/small_v3large_384_long/dog_face_landmarks_384_float16_static.tflite
cat dynamic  ~/PycharmProjects/cats-in-the-wild-ml/artifacts/small_v3large_384_long/cat_face_landmarks_384_float16.tflite
cat static   ~/PycharmProjects/cats-in-the-wild-ml/artifacts/small_v3large_384_long/cat_face_landmarks_384_float16_static.tflite
```

Reproduce with, in `~/PycharmProjects/dogs-in-the-wild-ml`:

```
scripts/reexport_static.py          # build the static variant, no retraining
scripts/bench_litert_macos.py       # XNNPACK latency, --use_xnnpack toggle in run()
scripts/run_delegate_matrix.py      # per-cell isolated engagement matrix
scripts/bench_metal.py              # Metal latency where it engages
scripts/pareto_harness.py           # full-split accuracy on the converted file
```

The cat model reproduces finding 1 (94.5 / 59.8 ms on 3.6.0, 28.2 / 59.9 on 3.7.0).

Also note **both face localizers** (`dog_face_localizer.tflite`,
`cat_face_localizer.tflite`) carry 23 `SHAPE` / 23 `STRIDED_SLICE` / 23 `PACK`, one
set per EfficientNetB2 block from stochastic-depth noise shapes, while every
`animal_detection` model (ssdlite, species classifier, rtmpose, hrnet) is already
fully static. A static re-export of the dog localizer measured 8.54 -> 8.32 ms under
XNNPACK with bit-identical output, so with no deconv to mis-delegate the graph shape
barely matters there. That is consistent with finding 1: the penalty was never about
shape as such, it was about which kernel ends up running `TRANSPOSE_CONV`.
