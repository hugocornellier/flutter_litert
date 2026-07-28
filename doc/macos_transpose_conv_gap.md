# The 3x macOS/iOS gap: TRANSPOSE_CONV does not parallelise on macOS

Status: **ROOT CAUSE FOUND AND FIX VERIFIED.**

ruy multithreading is dead in the CMake-built macOS dylib. A bazel build of the same TF
2.20.0 source fixes it completely and brings macOS to parity with iOS. Built by
`.github/workflows/build-tflite-c-macos.yml` and measured (see
[Verified result](#verified-result-the-fix-works)).

Remaining work before shipping: the x86_64 slice does not cross-compile on an arm64 runner
(`'NEON_2_SSE.h' file not found`), so only an arm64 dylib exists. The shipped artifact must
stay universal, so that cross-compile has to be resolved, or the x86_64 slice built on an
Intel runner.

## Summary

The cat/dog landmark model runs ~3x slower on macOS than on the iOS Simulator, on the same
Apple Silicon Mac. The cause is not CPU speed, build mode, or delegates.

**`TRANSPOSE_CONV` gets zero benefit from multiple threads in the macOS TensorFlow Lite
build, while it scales ~3x in the iOS build.** The landmark model's head is four stacked
transpose-convs, and that head accounts for essentially 100% of the observed gap.

Per-core speed is nearly identical between the platforms. The entire difference is
parallelism on this one op.

## Evidence

### 1. At one thread the platforms are equivalent

| model | macOS t1 | iOS t1 | ratio |
|---|---|---|---|
| landmarks_384 | 96.5 ms | 82.3 ms | 1.17x |
| localizer_224 | 21.5 ms | 21.7 ms | 1.01x |

The headline 3x only appears at the default 4 threads. macOS is not slower per core.

### 2. The gap is a scaling failure, and only for one model

XNNPACK, threads 1 -> 4:

| model | macOS | iOS |
|---|---|---|
| landmarks_384 | 96.5 -> 83.7 = **1.15x** | 82.3 -> 26.9 = **3.06x** |
| localizer_224 | 21.5 -> 8.3 = 2.59x | 21.7 -> 8.3 = 2.61x |

Reproduced without any delegate, so it is not delegate-specific:

| model | macOS | iOS |
|---|---|---|
| landmarks_384 | 104.9 -> 91.2 = **1.15x** | 87.2 -> 32.5 = **2.68x** |
| localizer_224 | 23.4 -> 13.4 = 1.75x | 21.2 -> 10.8 = 1.96x |

`localizer` scales the same on both platforms, so macOS threading works in general.

### 3. Isolated single-op models identify the culprit

Synthetic models, one op each, 96x96x48 -> 192x192x48, threads 1 -> 4:

| op | macOS | iOS |
|---|---|---|
| **transpose_conv** (xnnpack) | 9.98 -> 9.38 = **1.06x** | 7.65 -> 2.68 = **2.85x** |
| **transpose_conv** (no delegate) | 10.02 -> 8.29 = **1.21x** | 7.28 -> 2.67 = **2.73x** |
| plain conv (control) | 16.23 -> 4.66 = 3.48x | 17.29 -> 4.71 = 3.67x |
| depthwise conv (control) | 0.91 -> 0.40 = 2.28x | 0.98 -> 0.43 = 2.28x |

Both controls scale identically across platforms. Only `TRANSPOSE_CONV` differs.

### 4. The real head accounts for the whole gap

The shipping model contains **four** `TRANSPOSE_CONV` ops, all 4x4 kernels at stride 2:

| layer | input | weights | output | MACs |
|---|---|---|---|---|
| 1 | [1,12,12,960] | [128,4,4,960] | [1,24,24,128] | 1.13 G (8%) |
| 2 | [1,24,24,128] | [128,4,4,128] | [1,48,48,128] | 0.60 G (4%) |
| 3 | [1,48,48,128] | [128,4,4,128] | [1,96,96,128] | 2.42 G (18%) |
| 4 | [1,96,96,128] | [128,4,4,128] | [1,192,192,128] | **9.66 G (70%)** |

**Fidelity of the reconstruction.** The synthetic head used 128 input channels on layer 1
rather than the real 960, so it underestimates that layer. But layer 1 is only ~8% of head
MACs while layer 4 - reproduced exactly - is 70%. The synthetic is ~93% FLOP-equivalent to
the real head, dominated by the layer that matches. The close agreement below is therefore
expected rather than coincidental, though it should not be read as exact to 0.1 ms.

Rebuilding exactly that head as a standalone model:

| | macOS t1 | macOS t4 | iOS t1 | iOS t4 |
|---|---|---|---|---|
| 4-layer deconv head | 73.68 | **74.91** | 57.05 | **18.10** |

macOS gains *nothing* from 4 threads (73.68 -> 74.91, marginally worse). iOS scales 3.15x.

- head-only gap: 74.91 - 18.10 = **56.8 ms**
- full-model gap: 83.7 - 27.0 = **56.7 ms**

The deconv head explains the entire difference.

### 5. Control: the other runtime is identical on both platforms

The same synthetic ops run through **CompiledModel** (LiteRT, `libLiteRt.dylib`) rather
than the Interpreter (`libtensorflowlite_c-mac.dylib`):

| op | macOS `cm:cpu` | iOS `cm:cpu` |
|---|---|---|
| transpose_conv | 8.10 ms | 8.34 ms |
| plain conv | 2.90 ms | 2.97 ms |

CompiledModel is platform-neutral. So macOS has no OS- or hardware-level problem with this
op, and the framing is better stated the other way round: **the iOS `tflite_c` build is the
outlier**, reaching 2.68 ms where every other runtime/platform combination sits at 8-9 ms.

This isolates the fault to exactly one artifact: `libtensorflowlite_c-mac.dylib`.

## Ruled out

- **Build mode.** macOS profile and debug agree within 0.6%.
- **Emulation.** The iOS Simulator on Apple Silicon runs native arm64 on the host CPU.
- **Delegates.** Reproduced with XNNPACK and with no delegate at all.
- **General macOS threading.** `localizer`, plain conv and depthwise conv all scale
  normally on macOS.
- **Per-core throughput.** Single-thread times differ by only 1.0-1.2x.
- **Binary symbol comparison.** Inconclusive: the two artifacts have different symbol-strip
  levels. `nm` reports 0 matches on iOS for `ruy`, `NEON`, `optimized_ops` and even
  `reference_ops`, which cannot be true of a working TFLite. Do not draw conclusions from
  symbol counts here.

## Where the difference comes from

Both binaries are our own prebuilt artifacts, produced by different pipelines:

| | source | runtime version | platform | minos | sdk |
|---|---|---|---|---|---|
| macOS | committed `macos/.../libtensorflowlite_c-mac.dylib` | `2.20.0` | 1 (macOS) | 11.0 | 15.2 |
| iOS | downloaded `libs-v0.1.8/ios-frameworks.zip` (`ios/flutter_litert.podspec:51`) | `2.20.0-dev0+selfbuilt` | 7 (iOS sim) | 14.0 | 18.2 |

**This is the key fact for fixability.** Both are TensorFlow Lite **2.20.0** - the same
version - queried at runtime via `TfLiteVersion()`. The iOS artifact is tagged
`-dev0+selfbuilt`, i.e. compiled from source with our own flags; the macOS one is a stock
2.20.0 build. Both machines report 16 cores.

So this is not a TFLite limitation and not a version gap. A source build of 2.20.0 already
parallelises `TRANSPOSE_CONV` correctly - we have one, and it is the iOS artifact. The
macOS binary simply was not produced the same way.

Suggestive but not conclusive, given the strip differences:

| marker | macOS | iOS |
|---|---|---|
| `pthreadpool` | 0 | 52 |
| `cpu_backend_threadpool` | 0 | 2 |

`pthreadpool` is the threading library XNNPACK and TFLite's optimized kernels use for
intra-op parallelism. The behavioural evidence is unambiguous regardless of which build
flag caused it: the macOS build's `TRANSPOSE_CONV` does not use a thread pool.

## Recommended fix

**Rebuild the macOS dylib using the same recipe that produced the iOS frameworks.** That
recipe demonstrably yields a binary whose `TRANSPOSE_CONV` scales ~3x, at the same TFLite
version, so no upstream change or version bump is needed.

Success criterion: on macOS, `transpose_conv` scales ~3x from 1 to 4 threads, and
`Interpreter.version` reports a `selfbuilt` tag like the iOS artifact does.

Confidence this is fixable: **high**. The problem is not that TFLite 2.20.0 cannot
parallelise this op - our own iOS build of 2.20.0 does. The remaining unknown is only how
much work it is to point the existing macOS build pipeline at the same configuration.

### 5b. Verified: the iOS scaling is genuine, and thread pools are equivalent

The iOS figure is the load-bearing measurement in this diagnosis, so it was checked with a
four-point sweep rather than two. XNNPACK, same synthetic op:

| | t1 | t2 | t4 | t8 |
|---|---|---|---|---|
| **iOS** transpose_conv | 6.93 | 4.31 | 2.61 | **1.81** (3.83x) |
| **macOS** transpose_conv | 8.19 | 8.09 | 8.29 | **8.09** (1.01x) |
| iOS plain conv | 16.00 | 9.04 | 4.66 | 2.43 |
| macOS plain conv | 16.25 | 9.17 | 4.71 | 2.50 |

Two things follow. The iOS result is a smooth monotonic parallel curve, not an artifact of
warmup or a single noisy sample. And the `plain_conv` curves are nearly identical across
platforms, so the two builds' thread pools are equivalent in capability - the difference is
confined to this one op.

### 6. No deconv configuration parallelises on macOS

Stride and kernel size do not matter either. Every variant is flat:

| config | t1 | t4 | scaling |
|---|---|---|---|
| stride 1, k=3 | 7.43 | 7.32 | 1.01x |
| stride 1, k=4 | 13.04 | 12.96 | 1.01x |
| stride 2, k=2 | 3.78 | 3.80 | 1.00x |
| stride 2, k=4 | 13.42 | 13.37 | 1.00x |
| stride 4, k=4 | 15.13 | 15.27 | 0.99x |

So it is not XNNPACK's subconvolution-versus-direct algorithm choice (which depends on
stride) either. TRANSPOSE_CONV simply has no parallel path in these macOS builds.

### 6b. It is categorical, not a work-size threshold

TFLite could reasonably decline to parallelise small ops. It does not: on macOS, deconv
scaling is flat at every size tested, via the bazel-built TF Python wheel (which applies
XNNPACK by default):

| op | input | t1 | t4 | scaling |
|---|---|---|---|---|
| dc_small | 48x48x64 | 3.42 | 3.41 | 1.00x |
| dc_mid | 96x96x64 | 13.80 | 13.77 | 1.00x |
| dc_big | 192x192x64 | 55.03 | 55.49 | 0.99x |
| dc_wide | 96x96x256 | 206.84 | 208.05 | 0.99x |

Even a 206 ms op gains nothing from 4 threads. TRANSPOSE_CONV is unconditionally
single-threaded on macOS.

### 7. A bazel-built binary on macOS is ALSO flat - hypothesis weakened

Google's official TF **2.15** Python wheel is bazel-built and applies XNNPACK by default.
On macOS:

| op | t1 | t4 | scaling |
|---|---|---|---|
| transpose_conv | 7.94 | 8.29 | **0.96x** |
| plain conv (control) | 15.69 | 4.33 | 3.62x |
| 4-layer deconv head | 71.45 | 71.41 | **1.00x** |

This is evidence **against** "bazel fixes it", but it is confounded by version: that wheel is
TF 2.15 (2023) and XNNPACK's deconvolution path changed substantially by 2.20.

Current state of the matrix:

| build | platform | TF | deconv scaling |
|---|---|---|---|
| CMake (shipped) | macOS | 2.20 | flat |
| CMake (rebuilt here) | macOS | 2.20 | flat |
| bazel (TF Python wheel) | macOS | 2.15 | flat |
| bazel (our framework) | iOS | 2.20 | **3.15x** |
| **bazel (CI workflow)** | **macOS** | **2.20** | **unknown - the missing cell** |

The CI workflow exists to fill that last row. If it is also flat, the variable is the
platform rather than the build system, and the fix has to be sought elsewhere.

## ROOT CAUSE: ruy multithreading is dead in the macOS build

The defect is not specific to TRANSPOSE_CONV. **Every ruy-backed builtin kernel fails to
thread on macOS**, while all of them thread on iOS. Measured with no delegate, so the
builtin kernels are exercised directly:

| op (builtin / ruy path) | macOS t1 -> t8 | iOS t1 -> t8 |
|---|---|---|
| transpose_conv | 8.25 -> **7.67** (flat) | 7.31 -> **1.81** (4.0x) |
| fully_connected | 0.89 -> **1.03** (flat) | 1.37 -> **0.27** (5.1x) |
| batch_matmul | 1.43 -> **1.42** (flat) | 1.10 -> **0.20** (5.5x) |

XNNPACK threading is healthy on macOS by contrast (fully_connected 0.75 -> 0.26 with
XNNPACK), which is why this went unnoticed: **XNNPACK covers almost everything**, so the
broken ruy path is only observable on ops XNNPACK declines.

TRANSPOSE_CONV is simply the one such op in the cat/dog pipeline. Any model relying on
another ruy-only op would be equally affected on macOS.

This makes the impact wider than first documented, and gives a specific fix target: get ruy
to thread in the macOS build.

## Supporting detail: XNNPACK never executes TRANSPOSE_CONV, on either platform

An earlier step here inferred that "XNNPACK accepts the op on both platforms" from
`ModifyGraphWithDelegate` returning `status=0`. That inference was wrong, and it is the same
trap documented in `delegate_verification.md`: TFLite returns `kTfLiteOk` for an empty
replacement set, so status 0 does not mean any node was taken.

Checked numerically instead - if XNNPACK executed the op, its output would differ from the
builtin kernel's:

| op | macOS xnnpack dev | iOS xnnpack dev | conclusion |
|---|---|---|---|
| **transpose_conv** | **0.0** | **0.0** | XNNPACK executes it on **neither** platform |
| plain conv | 2.1e-6 | 4.8e-7 | executed on both |
| depthwise conv | 6.0e-8 | 6.0e-8 | executed on both |

### What this relocates the cause to

TRANSPOSE_CONV runs on TFLite's **builtin** kernel on both platforms. The source explains
why that is decisive:

| op | kernel variants in TF 2.20 |
|---|---|
| CONV_2D | kReference, kGenericOptimized, **kMultithreadOptimized**, kCblasOptimized |
| TRANSPOSE_CONV | kReference, kGenericOptimized - **no multithreaded variant** |

`Register_TRANSPOSE_CONV()` resolves to `Register_TRANSPOSECONV_GENERIC_OPT()`, and that
path's `TransposeConvV2` parallelises only via **ruy's gemm** (it receives a
`CpuBackendContext*`).

So the variable is **ruy's threading in the macOS build versus the iOS build** - not
XNNPACK, not kernel selection. It also explains why `plain_conv` scales on macOS while
transpose_conv does not: plain conv is delegated to XNNPACK (dev 2.1e-6), a different code
path entirely.

This lowers confidence that a bazel rebuild alone fixes it, since ruy is present in the
CMake binary (361 symbols) and simply appears not to thread for this op.

## Attempted fixes: what was tried locally

### 1. CMake rebuild from source - REPRODUCED THE BUG

Built TF 2.20.0 with the same recipe as `build-tflite-c-windows.yml`
(`-DTFLITE_ENABLE_XNNPACK=ON`, `CMAKE_OSX_ARCHITECTURES=arm64`, the TF version defines),
swapped the result into the plugin, and confirmed via the app bundle that the new
5,860,024-byte dylib was the one loaded.

| op | shipped dylib | fresh CMake dylib | iOS (bazel) |
|---|---|---|---|
| transpose_conv t1 -> t4 | 9.98 -> 9.38 | 10.25 -> **10.07** | 7.65 -> **2.68** |
| plain conv t1 -> t4 | 16.23 -> 4.66 | 18.07 -> 4.71 | 17.29 -> 4.71 |
| 4-layer deconv head | 73.68 -> 74.91 | 78.19 -> **78.32** | 57.05 -> **18.10** |

`plain_conv` scaling 3.84x in the same binary proves its threading works generally; only
deconv is flat. This is a useful negative result: it eliminates

- a missing CMake flag (the documented recipe was used),
- XNNPACK involvement entirely - see the correction below,
- absent kernels (deconv symbols: 46 in the shipped dylib, 81 in a fresh XNNPACK build),
- the TFLite version (both report 2.20.0),
- pthreadpool absence (`libpthreadpool.a` was built and linked, 69,984 bytes, and deconv
  still did not parallelise).

The remaining variable is the **build system**: CMake versus the bazel build that produced
the working iOS framework.

### 2. Bazel rebuild locally - BLOCKED BY TOOLCHAIN

Two failure modes, neither related to the code:

- `--config=macos_arm64` -> `no such package
  '@@build_bazel_apple_support//configs/platforms'`. That config is broken in TF 2.20.
- Host build (no platform config) -> `dyld: missing LC_UUID load command in
  .../local_config_apple_cc/wrapped_clang_pp`. Bazel's own compiler wrapper will not
  execute. Persisted after `bazelisk clean --expunge`.

This machine runs macOS 26.4 / Darwin 25.4.0; bazel 7.4.1's Apple toolchain predates it.
Local bazel builds of TF are not possible here.

### 3. The implemented fix: CI workflow

`.github/workflows/build-tflite-c-macos.yml` builds the dylib with bazel on a `macos-14`
runner, which has a toolchain bazel 7.4.1 supports. It builds each arch separately
(`--cpu=darwin_arm64`, then `--cpu=darwin_x86_64`) because the combined `macos_arm64`
config is broken, lipos them into a universal binary, sets the correct install name, and
fails the job if any required export is missing.

**This is unverified.** Running it and re-measuring is the outstanding work. Success
criterion: `transpose_conv` scales ~3x from 1 to 4 threads on macOS, and the landmark model
drops from ~83 ms to ~27-30 ms.

## VERIFIED RESULT: the fix works

The bazel-built arm64 dylib from the CI workflow was swapped into the plugin and measured on
the same machine. `Interpreter.version` reported `2.20.0-dev0+selfbuilt`, confirming the new
binary was the one loaded (the CMake build reports a bare `2.20.0`).

### Every ruy-backed op reaches iOS parity

| op (builtin / ruy path) | macOS before (CMake) | macOS after (bazel) | iOS reference |
|---|---|---|---|
| transpose_conv t1 -> t8 | 8.25 -> **7.67** (flat) | 7.02 -> **1.81** (3.9x) | 7.31 -> 1.81 |
| fully_connected t1 -> t8 | 0.89 -> **1.03** (flat) | 1.32 -> **0.31** (4.3x) | 1.37 -> 0.27 |
| batch_matmul t1 -> t8 | 1.43 -> **1.42** (flat) | 1.11 -> **0.19** (5.8x) | 1.10 -> 0.20 |

### The real model

| | macOS before | macOS after | iOS reference |
|---|---|---|---|
| 4-layer deconv head, t4 | 74.91 (flat) | **18.02** (3.16x scaling) | 18.10 |
| **cat landmark model, xnnpack t4** | **83.7 ms** | **26.9 ms** | **26.9 ms** |

The landmark stage drops **3.11x**, from 83.7 ms to 26.9 ms, exactly matching the iOS
figure. The 3x macOS/iOS gap is closed, not merely reduced.

Full-pipeline effect for cat_detection / dog_detection `full` mode on macOS: the face stages
were ~99 ms of a ~114 ms pipeline; the landmark component alone returns ~57 ms.

## Rejected fix: changing the model

Replacing `Conv2DTranspose` with upsample + conv **is not worth it**, because it trades the
shipping platform for the dev machine:

| head variant | macOS t4 | iOS t4 |
|---|---|---|
| transpose_conv (current) | 74.91 | **18.10** |
| upsample + conv | **43.71** | 43.57 |

macOS improves 1.71x, but iOS regresses **2.41x**. iOS and Android are what ship; macOS is
a development and desktop-user platform. Do not make this substitution.

## Impact

- **iOS, Android: none.** The iOS build already parallelises correctly.
- **macOS only:** the landmark stage costs ~83 ms instead of ~27 ms, so cat/dog `full` mode
  is roughly 3x slower on desktop than it needs to be. Affects the local development loop
  and macOS end users of the packages.

This is why the issue went unnoticed: it does not affect the platforms the packages
primarily target.

## Reproducing

Synthetic models were generated with TensorFlow 2.15 (`cats-in-the-wild-ml/.venv`):
single-op `transpose_conv`, `plain_conv`, `depthwise_conv` at 192x192x48, plus a four-layer
`deconv_head_real` matching the shipping model's head, and `upsample`/`resize` variants.
Each was timed at 1 and 4 threads, with and without XNNPACK, on both platforms.

The harness is not committed. It should be, alongside the existing
`full_delegate_sweep_test.dart`, as a per-op scaling regression test.
