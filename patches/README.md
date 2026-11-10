# Patches applied to TensorFlow source before building delegate binaries

Each patch here is applied by the corresponding `.github/workflows/build-*.yml`
after the TensorFlow clone and before `configure.py`. Keep them minimal and
document why, because they have to be rebased on every `TF_VERSION` bump.

## litert_coreml_npu_ios.patch

Extends the existing iOS `TensorFlowLiteCCoreML` framework with a dedicated
entry point for LiteRT `CompiledModel` requests containing `Accelerator.npu`.
The ordinary Interpreter delegate remains on `MLComputeUnitsAll`; the new
entry point uses `MLComputeUnitsCPUAndNeuralEngine`, which excludes Metal while
allowing Core ML to use the Neural Engine and CPU.

The patch also exports a thread-local delegated-node counter. The iOS
accelerator-registration bridge reads it after compilation and rejects a
zero-node Core ML result rather than silently returning CPU-only inference.
It is deliberately separate from `coreml_mean_padding.patch`; the iOS build
workflow applies both patches to TensorFlow v2.21.0.

The matching workflow is `build-coreml-ios.yml`. It builds an arm64 device
slice and a universal arm64+x86_64 simulator slice, then verifies the new NPU
symbols before assembling `TensorFlowLiteCCoreML.xcframework`. Simulator tests
validate integration only because an iOS simulator has no Apple Neural Engine.

## litert_coreml_npu_macos.patch

Builds a dedicated macOS arm64 Core ML delegate for LiteRT
`Accelerator.npu`. The delegate uses `MLComputeUnitsCPUAndNeuralEngine`, which
excludes GPU execution while allowing Core ML to keep unsupported operations on
the CPU. It is separate from the classic Interpreter Core ML delegate, whose
upstream `MLComputeUnitsAll` behavior is unchanged.

The patch also carries the existing required-padding fix for global `MEAN`
pooling. (The protobuf `RepeatedField::Resize` fix it carried for v2.1.5 is
upstream since v2.3.0, so it was dropped.) A thread-local diagnostic
counter records how many TFLite nodes the most recent Core ML delegate claimed,
allowing mixed NPU+CPU compilation to reject a silent zero-op fallback. The
matching workflow is `build-coreml-npu-macos.yml`; it pins LiteRT v2.3.2 at commit
`d30595a09f74b4c4c4f669eafeb26928a92deb5a` and produces both the delegate and
the small LiteRT accelerator-registration bridge.

## gpu_transpose_conv_v4.patch

Lets the GPU delegate accept `TRANSPOSE_CONV` version 4.

**Why.** Keras folds `Conv2DTranspose -> BatchNormalization -> ReLU` into a single
`TRANSPOSE_CONV` carrying a fused activation. Per TF's own header, that fused
activation *is* what version 4 adds:

```c
typedef struct {
  // Parameters supported by version 1:
  TfLitePadding padding;
  int stride_width;
  int stride_height;

  // Parameters supported by version 4:
  TfLiteFusedActivation activation;
  ...
} TfLiteTransposeConvParams;
```

`TransposeConvBuiltinOperationParser::IsSupported` caps at version 3, so the
delegate refuses the op with:

```
TRANSPOSE_CONV: Max version supported: 3. Requested version 4.
```

Every deconv-headed model exported from Keras hits this: segmentation, pose,
heatmap regression. `CheckGpuDelegateCompatibility` does *not* gate on version
for this op (it only checks strides and I/O), so the parser's cap is the sole
blocker.

**What it does.** Two hunks in
`tensorflow/lite/delegates/gpu/common/model_builder.cc`: raise the gate to 4, and
apply the activation via `MaybeFuseActivation`, exactly as
`Conv2DOperationParser` already does. Both helpers are already linked into the
shipped dylib. Without the second hunk the delegate would silently drop the
activation and return wrong results, which is why the gate cannot simply be
raised on its own.

**Measured payoff.** On the dog face landmark model (MobileNetV3Large + a
4-deconv heatmap head, 384px), macOS arm64 M4 Max, flutter_litert 3.7.0, over all
480 DogFLW test images:

| configuration | median `invoke()` | NME_IOD |
|---|---|---|
| XNNPACK, as shipped | 26.83 ms | 8.5664 |
| Metal, deconv rejected, falls to CPU | 30.25 ms | 8.5664 |
| Metal, deconv accepted | **5.11 ms** | 8.5665 |

The third row was obtained by rewriting the model to move the ReLU into a
separate `RELU` op, which drops the opcode to version 3 and is numerically
identical (max per-coordinate difference 6.6e-07 against the shipped model). That
rewrite is a workaround for exactly this gate. With the patch, unmodified Keras
exports get the same result and the rewrite becomes unnecessary.

So the patch is **de-risked in advance**: the GPU kernels are already known to
compute this graph correctly and quickly. The only open question is parser
plumbing.

**Status: VERIFIED.** `build-metal-macos.yml` run 30503901056 built it against
v2.20.0 and the resulting dylib was measured over all 480 DogFLW test images:

| model | Metal, stock dylib | Metal, patched dylib | NME_IOD |
|---|---|---|---|
| static export, v4 with fused ReLU | 30.20 ms | **5.18 ms** | 8.5665 |
| static export, ReLU already unfused (v3) | 5.17 ms | 5.17 ms | 8.5665 |
| shipped dynamic-shape export | creation fails | creation fails | -- |

Three things that result establishes:

1. The unmodified v4 asset now reaches 5.18 ms, matching the hand-rewritten v3 one.
   **The flatbuffer rewrite is therefore unnecessary on a patched dylib.**
2. Accuracy is correct. This is the load-bearing check, not the latency: had the
   version gate been raised without `MaybeFuseActivation` firing, the ReLU would
   have been dropped silently and NME would have collapsed rather than holding at
   8.5665. Both hunks work.
3. The v3 row is unchanged on stock versus patched, so the patch does not disturb
   the path that already worked.

The dynamic-shape export still fails GPU interpreter creation on both. That is a
separate blocker (a runtime-shaped tensor, not an op version) which this patch does
not address and is not meant to.

**Scope.** Wired into `build-metal-macos.yml`, which builds
`//tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib` for macOS. iOS needs
the same patch applied to whatever produces `TensorFlowLiteCMetal`, since
`metal_delegate_native.dart` resolves Metal from `tfliteBinding` on iOS rather
than the separate macOS dylib. `build-flex-ios.yml` shows the from-source iOS
pattern (bazel, same `TF_VERSION`), so this is reachable, but there is no
existing workflow building the Metal framework specifically. **iOS is the
platform where this matters most, because `InterpreterFactory` auto-mode selects
GPU there.**

## coreml_mean_padding.patch

Lets the CoreML delegate compile models whose supported global-spatial `MEAN`
ops reach its pooling builder. It does not add support for other `MEAN` forms,
such as non-4D inputs or reductions over dimensions other than height and
width.

**Why.** `PoolingLayerBuilder::Build()` handles `MEAN` as global average pooling and
returns early:

```cpp
if (pooling_type_ == kTfLiteBuiltinMean) {
  pooling_params->set_type(AVERAGE);
  pooling_params->set_globalpooling(true);
  return layer_.release();          // returns before the padding block
}
...
if (params->padding == kTfLitePaddingSame) {   // never reached for MEAN
  pooling_params->mutable_same();
} else {
  pooling_params->mutable_valid();
}
```

`PoolingLayerParams.PaddingType` is a required oneof, so leaving it unset makes Core
ML's validator reject the entire model at compile time:

```
validator error: Padding type for the pooling layer 'PoolingLayerBuilder (MEAN)_32' is not set.
ERROR: Failed to Compile and save Model.
ERROR: CoreMl Kernel was not initialized
ERROR: Node number 279 (TfLiteCoreMlDelegate) failed to prepare.
ERROR: Restored original execution plan after delegate application failure.
```

TFLite then restores the CPU plan and the model runs, so there is no crash and no
error surfaced to the caller. The delegate simply never does anything.

**Blast radius.** Any model with a `MEAN` op, which includes every MobileNetV3 and
EfficientNet variant (squeeze-excite blocks and global pooling). Confirmed on three
structurally different models here: `species_classifier_float16`, and both the
dynamic and static dog landmark exports.

**Measured cost of the bug.** On an iPhone 15 Pro, `PerformanceMode.coreml` is not
merely inert, it is worse than the default, because falling back to bare CPU loses
XNNPACK:

| landmark model, iPhone 15 Pro | median invoke | deviation from CPU |
|---|---|---|
| XNNPACK | 47.41 ms | 7.0e-06, engaged |
| CoreML | 57.38 ms | **0.0, no-op** |

**What it does.** Sets `mutable_valid()` before the early return. Global pooling
covers the whole spatial extent, so VALID is the correct padding type.

**Status: VERIFIED on both macOS and iOS device.** The fix works. It does not make Core
ML fast.

macOS (M4 Max), dog landmark static export, all 480 DogFLW test images:

| | stock dylib | patched dylib |
|---|---|---|
| outcome | interpreter creation fails | engaged |
| median invoke | -- | 27.51 ms |
| NME_IOD | -- | 8.5660 (CPU reference 8.5664) |
| deviation from CPU | -- | 5.6e-03, fp16 scale |

iPhone 15 Pro, via `build-coreml-ios.yml` and the SPM path override:

| landmark variant | backend | median | deviation | note |
|---|---|---|---|---|
| dynamic (ships today) | xnnpack | **47.82 ms** | 7.0e-06 | engaged |
| dynamic (ships today) | coreml | 57.92 ms | **0.0** | still a no-op |
| static, v4 fused ReLU | **coreml** | **47.89 ms** | **9.5e-04** | **engaged** |
| static, ReLU unfused | coreml | 56.29 ms | 8.9e-04 | engaged |
| static, ReLU unfused | gpu | 46.65 ms | 3.5e-05 | engaged |

Two conclusions, and they point in opposite directions:

1. **The fix is real.** Core ML went from deviation exactly 0.0 (attached, delegated
   nothing) to 9.5e-04 on device. The ANE is genuinely running the model, and the
   deviation is fp16 rounding rather than corruption. The dynamic export stays a no-op
   because its runtime-shaped tensors are a separate blocker this patch does not touch.
2. **It buys no speed.** 47.89 ms against XNNPACK's 47.82 ms is a tie. So is the GPU at
   46.65 ms. Three different compute units land within 2% of each other, which suggests
   this model is memory-bandwidth bound on an A17 Pro rather than compute bound. That
   also explains why 40 GPU cores on an M4 Max reach 5.11 ms while 6 cores plus an ANE
   both converge near 47 ms here.

So the value of this patch is not performance, it is **removing a silent regression**.
Before it, selecting `PerformanceMode.coreml` cost 57.92 ms against the 47.82 ms default,
because failing to compile drops the model to bare CPU without XNNPACK. Same shape as the
GPU no-op. After it, Core ML is at least an honest tie.

For these models the recommendation remains XNNPACK.

**Scope.** Wired into `build-coreml-macos.yml`, which builds
`//tensorflow/lite/delegates/coreml:tensorflow_lite_coreml_dylib` for macOS. iOS pulls
`TensorFlowLiteCCoreML` from a release instead; `build-coreml-ios.yml` builds that
framework from the same TensorFlow version with this patch applied.

**Packaged macOS artifact.** The dylib committed at
`macos/flutter_litert/Sources/flutter_litert/Resources/libtensorflowlite_coreml-mac.dylib`
was rebuilt from TensorFlow `v2.21.0` by
[workflow run 42666219391](https://github.com/hugocornellier/flutter_litert/actions/runs/42666219391)
with this patch applied. Its SHA-256 is
`7b2cc71c071036c9bb7fc7f13a089bbdd03c4d1afaea661364a19a9dbd56a937`.
The previous `v2.20.0` build, from
[workflow run 30707513086](https://github.com/hugocornellier/flutter_litert/actions/runs/30707513086),
had SHA-256 `c33c4e904613aed37a409b2805db7099edc4a8952f3e4794f5e84e58c5654580`.

On that `v2.20.0` dylib, the post-build macOS matrix exercised all 29 published
models through the classic CoreML delegate. The four models that previously failed Core ML
validation now compile and execute: `species_classifier_float16`,
`superanimal_rtmpose_s_float16`, `superanimal_ssdlite_float16`, and
`selfie_segmenter_landscape`. CPU-reference parity passes for SSD-Lite; the
other three remain accuracy-policy failures and must not be treated as safe
CoreML choices. `coreml_delegate_test.dart` uses the species classifier to
assert that this global-MEAN path stays delegated instead of silently retrying
on CPU.

## apple_support_objc_compile_actions.patch

Makes the Apple C/C++ toolchain of TensorFlow v2.21.0 pass its compile flags
and Xcode environment to Objective-C and Objective-C++ actions. Unlike the
other patches here, it patches Bazel's `apple_support` dependency, not the
TensorFlow tree.

**Why.** TF v2.21.0 moved to `apple_support` 1.24.5 (v2.20.0 used 1.18.1 and did
not hit this). Its `crosstool/cc_toolchain_config.bzl` scopes `-c <source>`,
`-target ...` and the `XCODE_VERSION_OVERRIDE` / `APPLE_SDK_*` environment to
`ACTION_NAME_GROUPS.all_cc_compile_actions`. Only `rules_cc` 0.2 and newer put
`objc-compile` and `objc++-compile` in that group, and TensorFlow's WORKSPACE
resolves an older `rules_cc`. So Bazel never sets `DEVELOPER_DIR` for those
compiles, and the first `.mm` file (`gpu/metal/common.mm`,
`coreml/coreml_executor.mm`) fails with:

```
Error: DEVELOPER_DIR not set.
```

The same applies to `objc-executable`, the `ObjcLink` action of `macos_dylib`,
which `rules_cc` 0.2 lists in `cc_link_executable_actions` and older versions
do not.

**What it does.** Adds `objc-compile` and `objc++-compile` to the compile action
list, and `objc-executable` to the dynamic link action list, wherever they are
missing, as `rules_cc` 0.2.13 (the version `apple_support` 1.24.5 requires)
already does. With a newer `rules_cc` the additions are no-ops.

**How it is applied.** `scripts/patch_tf_apple_support.sh <tf-src>` copies the
patch into the checkout's `third_party/` and adds it as `patch_file` to the
`build_bazel_apple_support` archive in `tensorflow/workspace2.bzl`, so Bazel
patches `apple_support` when it fetches it. The archive's SHA-256 is still
checked, and the script fails if the checkout does not fetch `apple_support`
1.24.5 at the pinned SHA-256. It runs after the TensorFlow clone in
`build-tflite-ios.yml`, `build-coreml-ios.yml`, `build-metal-macos.yml`,
`build-coreml-macos.yml` and `build-flex-ios.yml`, which all compile
Objective-C(++) sources.

## pthreadpool_apple_spin.patch

Makes XNNPACK's thread pool on Apple platforms keep its workers awake between
the parallel sections of one inference, as the Grand Central Dispatch backend
did. Like `apple_support_objc_compile_actions.patch` it patches a Bazel
dependency (`pthreadpool`, which TensorFlow also fetches for the CMake build),
not the TensorFlow tree.

**Why.** TF v2.21.0 pins pthreadpool `0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e`,
which removed the Grand Central Dispatch backend (`src/gcd.c`) that v2.20.0's pin
used on iOS and macOS. Apple builds now use `src/pthreads.c`: a worker that
finds no work spins `PTHREADPOOL_SPIN_PAUSE_ITERATIONS` times and then waits on
a condition variable, and the main thread has to signal it for the next
parallel section. `src/threadpool-common.h` sets that count to 100000 on
Android and 1000 everywhere else, so on Apple the workers fall asleep between
the operators of one inference and every operator pays a wake-up. On a
physical iPhone 15 Pro with 4 XNNPACK threads this made the `Interpreter` in
4.1.0 a median 11% slower than 4.0.0 over 28 models (total time +8%), while
the same models at 1 thread were unchanged (median ratio 1.000), which is what
isolates the thread pool. See the 4.1.0 "Known issues" and the 4.1.1 entry in
`CHANGELOG.md` for the numbers.

**What it does.** Three things, all only on the condition-variable backend
(`PTHREADPOOL_USE_FUTEX 0`, which is what Apple builds use; the futex and
Windows paths are untouched).

1. `src/pthreads.c`: the three wait functions (`wait_on_num_recruited_threads`,
   `wait_on_num_active_threads`, `wait_on_work_is_done`) spin without the pool
   mutex. Upstream locks the mutex before the spin loop and holds it while
   spinning, and the signalling side (`signal_num_active_threads`,
   `signal_work_is_done`, `signal_num_recruited_threads`) locks the same mutex
   to broadcast. So only one thread can spin at a time, the others block in
   the kernel on the mutex, and the main thread's broadcast waits for the
   spinner. With the upstream count of 1000 that costs little; with a longer
   spin it serialises the pool (a first version of this patch that only raised
   the count made the macOS probe models 2 to 5 times slower). The patch spins
   on the atomic alone, and takes the mutex only to sleep: lock, re-check the
   atomic, `cond_wait` if it is still unmet, unlock. The signallers update the
   atomic before they take the mutex, so the re-check under the lock cannot
   miss a wake-up.
2. `src/pthreads.c`, `wait_on_work_is_done`: the main thread spins on a plain
   acquire load of `work_is_done` and exchanges it back to 0 only once it has
   seen it set. Upstream (and a second version of this patch) did the exchange
   on every spin iteration. The last worker also finishes a section with an
   exchange on that flag, and on the iPhone 15 Pro the worker lost that
   contention for the whole spin: with the lock-free waits and a 100000 spin,
   every parallel section took as long as the spin and the models ran 20 times
   slower, while the same build was the fastest on the Mac. With the plain
   load the phone runs the same 28 models a median 23 to 27% faster than the unpatched 4.1.0 build (total time 7 to 11% lower) and 13 to 18% faster than 4.0.0 (total 3 to 9% lower), 4 threads, two passes each; the 10000 spin is nearly as good (median 15 to 16% faster than unpatched), so the limit is not critical once the waits are plain loads.
3. `src/threadpool-common.h`: an `#elif defined(__APPLE__)` branch with
   `PTHREADPOOL_SPIN_PAUSE_ITERATIONS 100000` (overridable with
   `-DPTHREADPOOL_SPIN_PAUSE_ITERATIONS=N`) and `PTHREADPOOL_SPIN_YIELD_ITERATIONS 0`,
   so Apple spins as long as Android before a worker sleeps.

Each spin iteration is one load and one `yield` instruction (100000 of them
take about 60 microseconds on Apple silicon), so a worker that finds no more
work spins for well under a millisecond after the last parallel section and
then sleeps as before. CPU use between inferences is unchanged beyond that.

**How it is applied.** `scripts/patch_tf_pthreadpool.sh <tf-src>` copies the
patch into the checkout's `third_party/` and registers it twice: as
`patch_file` on the `pthreadpool` archive in `tensorflow/workspace2.bzl` for the
Bazel builds (iOS frameworks, macOS arm64 dylib), and as a `PATCH_COMMAND` on
the `ExternalProject` in `tensorflow/lite/cmake/DownloadPThreadPool.cmake` for
the CMake build (macOS x86_64 dylib). The archive's SHA-256 is still checked,
and the script fails if the checkout pins a different pthreadpool commit, so a
`TF_VERSION` bump has to re-check the patch. It runs after the TensorFlow clone
in `build-tflite-ios.yml` and in both compile jobs of `build-tflite-c-macos.yml`.
Linux and Windows builds are not patched: they keep the futex and Windows
backends, which the GCD removal did not change.
