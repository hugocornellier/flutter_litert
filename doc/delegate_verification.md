# Delegate verification: two silent-failure bugs

Status: investigated, not yet fixed. Detection mechanism **resolved** for GPU (see
[Spike result](#spike-result-resolved)). Independently reviewed and revised; see
[Merged review](#merged-review-independent-adversarial-analysis) at the end, which
supersedes earlier sections where marked.

A third pass re-verified every code and symbol claim against the tree. It corrected the
merged review's item #8 (`LiteRtGetStatusString` is not available on all platforms, so it
does not supersede bug 3), widened the teardown fix from two files to four sites, bounded
what the `ModifyGraphWithDelegate` probe can promise, and labelled the deviation tables by
input regime. Those changes are inline, marked where they overturn something.

## Summary

flutter_litert has two distinct silent-failure modes. Neither surfaces any error, and
neither is detectable from timing alone.

| # | API | Failure | Cost |
|---|-----|---------|------|
| 1 | `Interpreter` | Delegate attaches, delegates **zero ops**, runs on bare CPU. No warning. | 17-23% latency, silently |
| 2 | `CompiledModel` | Returns **numerically wrong output** for 3 of 5 models, both platforms, **including CPU-only**. No exception. | Wrong answers at high confidence |

Bug 2 is the more serious of the two: bug 1 costs speed, bug 2 costs correctness.

Bug 1 is ours to fix. **Bug 2 is not**: it reproduces through Google's own CompiledModel
binding over the same LiteRT build, so the fix is upstream and the local action is to keep
`Interpreter` on the affected models. See
[Bugs 2 and 4: root cause is upstream LiteRT](#bugs-2-and-4-root-cause-is-upstream-litert-not-this-wrapper).

Both were found by comparing output against a CPU reference. Timing cannot distinguish
"ran on GPU and was slow" from "never ran on GPU at all", which is why they have gone
unnoticed.

## What is NOT broken

Recorded explicitly, because the investigation produced three wrong conclusions before
reaching the right one. Do not re-litigate these:

- **The Metal/GPU delegate works on macOS.** It accelerates MediaPipe's
  `face_detection_short_range` (dev 5.1e-4) and `face_landmark` (dev 2.0e-4) on the same
  machine where it no-ops on our models.
- **The CoreML delegate works.** It engages on the same two MediaPipe models
  (dev 7.6e-1 and 1.2e-1). An earlier reading of "0/10 non-functional" was an artifact of
  only testing our own models.
- **`_createCoreml` already passes `enabledDevices: 1` (`AllDevices`).** The
  Neural-Engine-only default in `CoreMlDelegateOptions` is already overridden, so that is
  not the cause.
- **Build mode is not a factor.** macOS profile and debug agree within 0.6%.

The delegates are fine. Specific *models* are not delegable, and the library does not say so.

## Evidence

Full matrix: 5 bundled models x 6 backends x 2 platforms. `dev` is max absolute deviation
from the same model's `interp:disabled` output. `dev = 0.0` means the backend produced
bit-identical output to plain CPU, i.e. it did nothing.

**Input regime: zeroed** (both tables in this section). Deviation figures elsewhere in this
document come from different runs under different input regimes and are not directly
comparable to these; each table below states its own.

### macOS

| model | interp:disabled | interp:xnnpack | interp:gpu | interp:coreml | cm:cpu | cm:gpu+cpu |
|---|---|---|---|---|---|---|
| ssdlite | 7.05 | 4.49 (1.4e-5) | 8.07 (2.4e-6) | 7.59 (**0.0**) | 4.92 (1.4e-5) | 6.56 (1.4e-5) |
| species_cls | 2.12 | 1.35 (1.3e-4) | 1.91 (**0.0**) | 1.96 (**0.0**) | 1.15 (1.4e-4) | 1.51 (**1.0e+1**) |
| rtmpose_s | 12.16 | 7.74 (7.0e-6) | 13.52 (2.6e-1) | 12.01 (**0.0**) | 5.62 (7.6e-6) | 13.52 (2.4e-1) |
| localizer_b2_224 | 13.80 | 8.33 (2.7e-7) | 13.82 (**0.0**) | 14.29 (**0.0**) | 11.11 (1.5e-7) | FAIL create=3 |
| landmarks_v3l_384 | 98.25 | 87.14 (2.3e-6) | 96.71 (**0.0**) | 95.45 (**0.0**) | FAIL run=3 | FAIL create=3 |

### iOS simulator

| model | interp:disabled | interp:xnnpack | interp:gpu | interp:coreml | cm:cpu | cm:gpu+cpu |
|---|---|---|---|---|---|---|
| ssdlite | 4.94 | 4.48 (2.6e-5) | 7.92 (3.8e-6) | 4.69 (**0.0**) | 4.73 (2.6e-5) | 7.75 (2.6e-5) |
| species_cls | 1.13 | 1.31 (1.4e-4) | 1.12 (**0.0**) | 1.23 (**0.0**) | 1.14 (1.5e-4) | 1.86 (**1.0e+1**) |
| rtmpose_s | 8.96 | 7.60 (7.8e-6) | 14.68 (2.6e-1) | 8.85 (**0.0**) | 5.58 (7.2e-6) | 11.04 (2.4e-1) |
| localizer_b2_224 | 10.48 | 8.06 (1.8e-7) | 10.51 (**0.0**) | 10.66 (**0.0**) | 11.03 (6.0e-8) | FAIL create=3 |
| landmarks_v3l_384 | 32.24 | 26.82 (2.3e-6) | 32.46 (**0.0**) | 32.96 (**0.0**) | FAIL run=3 | FAIL create=3 |

Times in ms, 8 iterations after 2 warmup. Zeroed inputs (conv nets are data-independent
in cost). Hardware: M-series Mac, 16 cores (12P/4E). iPhone 16 simulator, iOS 18.2.

**Input-degeneracy control.** Zeroed input risks false `dev = 0.0` readings if a network
saturates. Every finding above was re-run with deterministic pseudo-random input in [0,1]
and reproduced exactly, with reference-output spreads of 1.8e+1 (ssdlite), 1.0e+1
(species_cls), 8.4e-1 (rtmpose_s) and 4.1e-1 (landmarks) confirming the outputs are not
degenerate. An independent earlier run using a real cat photo and real localizer boxes also
reproduced the landmarks `gpu`/`coreml` no-op. Three input regimes, same conclusion.

### Which models delegate

- **GPU engages**: ssdlite, rtmpose_s
- **GPU no-ops**: species_cls, localizer (EfficientNetB2), landmarks (MobileNetV3Large)
- **CoreML no-ops**: all five

The three that fail are the EfficientNet / MobileNetV3-family models. Those use hard-swish
activations and squeeze-excite blocks, which are known gaps in TFLite's GPU delegate op
coverage. When a delegate cannot take a large contiguous partition it declines the graph
entirely, and says nothing.

## Bug 1: Interpreter delegates zero ops, silently

### Mechanism

`InterpreterOptions.addDelegate()` attaches the delegate before
`TfLiteInterpreterCreate`. Two outcomes are conflated:

1. `TfLiteInterpreterCreate` returns NULL. Already handled at
   `lib/src/native/interpreter.dart:90-113`: warns on stderr, retries without delegates.
   This path works correctly.
2. Creation succeeds but the delegate takes **zero ops**. `hasActiveDelegate` is set to
   `true`, no warning is emitted, and everything runs on bare CPU.

`hasActiveDelegate` currently means "a delegate was attached", not "a delegate is doing
anything".

### Consequence: `PerformanceMode.auto` is actively harmful on iOS

`InterpreterFactory._createAutoMode` picks GPU unconditionally on iOS. Confirmed by
measurement, not just source reading: on the simulator `auto` and `gpu` produce identical
deviations per model (ssdlite 1.9e-6 both, landmarks 0.0 both, species_cls 0.0 both).

For a non-delegable model that yields bare CPU, which is **slower than XNNPACK**:

| model | iOS auto (gpu, no-op) | iOS xnnpack | left on table |
|---|---|---|---|
| landmarks_v3l_384 | 32.46 | 26.82 | 17% |
| localizer_b2_224 | 10.51 | 8.06 | 23% |

XNNPACK is fastest in 6 of 10 model x platform cells. `cm:cpu` wins 3 (species_cls on
macOS, rtmpose on both) and GPU wins 1 (species_cls on iOS, where XNNPACK is in fact the
*slowest* backend). So this is not an argument for XNNPACK everywhere; it is an argument
for falling back to XNNPACK **when the requested accelerator does nothing**, which is a
narrower and better-supported claim.

### Fix

> **Superseded in part by the merged review.** `InterpreterFactory.create` is the wrong
> layer: it receives no model or interpreter and returns only
> `(InterpreterOptions, Delegate?)` (`interpreter_factory.dart:29`), so it cannot apply a
> delegate. The strategy below is right; it needs a new internal construct-apply-allocate
> path, not a change to `create`.

Try the requested accelerator, verify it engaged, fall back to XNNPACK (not bare CPU) when
it did not.

```
GPU path:    create -> ModifyGraphWithDelegate(gpu) -> status != 0 ?
                       -> ModifyGraphWithDelegate(xnnpack)
CoreML path: unchanged (create-time options.addDelegate)
```

**The probe must be applied to GPU only.** CoreML cannot be attached post-creation at all
(see the constraint section above), so routing it through `ModifyGraphWithDelegate` would
disable it wherever it currently works. Leave CoreML on the create-time path; if its
engagement ever needs verifying, that requires the numerical check.

Strictly better than today in every measured case: models that genuinely delegate keep GPU,
models that do not get XNNPACK instead of nothing, and CoreML is untouched.

Note the existing NULL-interpreter fallback at `interpreter.dart:90-113` stays reachable,
because CoreML (and any other create-time delegate) still goes through `options`. It does
not become dead code.

Emit a warning naming the requested and effective backend, so this is a console line rather
than a multi-hour investigation.

### Do not

Remove GPU from `auto` on iOS based on this data. The simulator's GPU is emulated and its
timings are not predictive of device. Verified fallback captures the win without making
that call. Confirm on hardware before changing the default accelerator choice.

## Bug 2: CompiledModel returns wrong numbers, silently

### Mechanism

`fromBufferWithGpuFallback` (`compiled_model_native.dart:196-224`) try/catches
**compilation** and falls back to CPU on a thrown error. That correctly handles the
undelegatable models, which throw `LiteRtCreateCompiledModel failed with LiteRtStatus=3`.

It does not handle the case where compilation succeeds and inference returns garbage:

**Input regime: real-valued** (necessarily so, since `landmarks` + `cm:cpu` throws under
zeroed input, per escalation 2 below. Confirm whether this was the deterministic
pseudo-random set or the cat-photo run and record it here.)

| model, config | deviation | output spread | error vs range |
|---|---|---|---|
| species_cls, `cm:gpu+cpu` | **8.3e+0** | 1.0e+1 | 83% |
| landmarks, **`cm:cpu`** | **6.2e-1** | 4.1e-1 | **151%** |
| rtmpose_s, `cm:gpu+cpu` | 3.4e-1 | 8.4e-1 | 40% |

> **Two cells need reconciling across sections.** `species_cls` on `cm:gpu+cpu` is reported
> as 1.0e+1 in the zeroed matrix, 8.3e+0 here, and 7.4e+0 in the precision/buffer-mode table;
> the matrix figure equals this table's *spread* column, which is what a transcription slip
> would look like. `rtmpose_s` on `cm:gpu+cpu` is 2.4e-1 in the matrix (both platforms),
> 3.4e-1 here, and 2.7e-1 / 2.5e-1 per-output in disputed item B, whose max should equal one
> of the other two and does not. Different input regimes explain some of this; confirm they
> explain all of it.
>
> Two apparent discrepancies that are **not** problems, recorded so they are not chased:
> `landmarks` + `cm:cpu` showing `FAIL run=3` in the matrix but a numeric deviation here is
> exactly the input-dependence finding below. And the 161% in the precision/buffer-mode table
> is arithmetically consistent with a reference spread near 3.98e-1, so it is a different run,
> not a bad division.

Identical values on macOS and the iOS simulator, reproduced across repeat runs. Deviations
are shown against the reference output's own spread: `landmarks` on `cm:cpu` is wrong by
more than the entire range of the correct output. These are unrelated answers, not drift.

Two escalations beyond the initial reading:

1. **Corruption is not GPU-specific.** `landmarks` is corrupted on **`cm:cpu`**, a CPU-only
   configuration with no accelerator involved. `fromBufferWithGpuFallback` cannot help
   here: there is no GPU to fall back from.
2. **The failure mode is input-dependent.** With *zeroed* input, `landmarks` + `cm:cpu`
   throws `LiteRtRunCompiledModel failed with LiteRtStatus=3`; with real-valued input it
   silently returns garbage. Both behaviours reproduce on both platforms. A smoke test
   using zero-filled tensors sees a clean exception and concludes the config is
   unsupported, while production traffic gets wrong numbers.

   > **Superseded: the variable is run count, not input values.** With *real-valued* input,
   > `landmarks` returns wrong output on run 1 and then throws
   > `LiteRtStatus=3` on **run 2** of the same instance. The earlier reading came from tests
   > that called `run()` once per instance, so the throw was never reached and the difference
   > got attributed to the input. Both symptoms occur under either input regime; zeroed input
   > only changes which one you notice first. The practical form of the bug is therefore
   > worse than recorded: the wrong numbers come first, and the exception arrives only after
   > a second inference on a reused instance.

So 3 of 5 models are numerically wrong under some CompiledModel configuration, and the
CPU-only case is the worst of the three.

A deviation of 10.0 on a classifier is a wrong label at high confidence. In
`animal_detection` this manifests as a cat classified `unknown_animal` at score 1.0000.
Deterministic, and identical on macOS (Metal) and the simulator.

### Root cause: neither Precision nor TensorBufferMode is the factor

All 4 combinations of `Precision` x `TensorBufferMode` were tested per accelerator set.
**Input regime: unrecorded.** Its `landmarks` and `species_cls` deviations differ from both
tables above, so this is a third run and needs its regime noted:

| model | accelerators | fp16/fp32 x managed/hostMemory | verdict |
|---|---|---|---|
| landmarks_v3l_384 | `{cpu}` | dev 6.4e-1 in **all 4** (161% of range) | corrupt, no workaround |
| landmarks_v3l_384 | `{gpu,cpu}` | create fails in **all 4** | unusable |
| species_cls | `{gpu,cpu}` | dev 7.4e+0 in **all 4** (75%) | corrupt |
| species_cls | `{cpu}` | dev 1.5e-5 in all 4 | **OK** |
| ssdlite (control) | both | OK in all 8 | harness sound |

Two consequences:

1. **There is no configuration workaround for `landmarks` under CompiledModel.** It is
   corrupt on CPU-only at both precisions and both buffer modes, and cannot compile at all
   for `{gpu,cpu}`. CompiledModel simply cannot run this model.
2. **For `species_cls` the corruption is GPU-specific**, and `{cpu}` is a correct fallback.
   So a verifier that detects corruption and retries CPU-only would fully fix this model,
   but not `landmarks`.

The control also validates the methodology: ssdlite at fp16 on `{gpu,cpu}` shows dev 1.2e-2
against 1.3e-5 at fp32 - a genuine ~1000x precision difference that is still 0% of the
output range. The check distinguishes fp16 drift from corruption rather than conflating
them.

### Note: CompiledModel is ahead of Interpreter here

For the no-op models CompiledModel **refuses loudly** where Interpreter degrades quietly.
The "detect and fall back" work is already done on this side. The gap is correctness, not
engagement.

### Fix

Extend the fallback trigger from "compilation threw" to "compilation threw **or** output is
wrong". Requires a numerical self-check (see below), wired into
`fromBufferWithGpuFallback` so a corrupt GPU result triggers the same CPU fallback an
exception already does.

## Bug 3 (minor): `LiteRtStatus` codes are unreadable

`compiled_model_native.dart:776` interpolates the raw integer:

```dart
error = 'LiteRtRunCompiledModel failed with LiteRtStatus=$status.';
```

Only `_kLiteRtStatusOk = 0` is defined anywhere in the codebase. Users see
`LiteRtStatus=3` with no way to learn it means a runtime failure. Add a code-to-name map.
Independent of everything else; cheap.

> **Not superseded by `LiteRtGetStatusString`.** The merged review originally proposed
> dropping this map in favour of binding the runtime's own status-string function. That
> symbol is not exported on every platform this package ships (see the table in step 1 of
> the suggested order), so the map is still needed as the fallback on Linux and Windows.
> Keep both: the bound function where it exists, this map where it does not.

## Bug 4: CompiledModel mishandles MobileNetV3Large + deconv

> **The title is a misattribution, kept for traceability.** Neither MobileNetV3Large nor deconv
> is the causal factor: the trigger is a **dynamic (runtime-shaped) model output**, and the
> first-run symptom is that the output buffer is never written at all. Bugs 2 and 4 are one
> upstream defect with a fix already in flight. See
> [Bugs 2 and 4: root cause is upstream LiteRT](#bugs-2-and-4-root-cause-is-upstream-litert-not-this-wrapper);
> read the measurements below as observations, not as diagnosis.

`landmarks_v3l_384` never produces a correct result under CompiledModel on either platform:

- `cm:gpu+cpu` -> always `LiteRtCreateCompiledModel failed with LiteRtStatus=3`
- `cm:cpu` -> with zeroed input, `LiteRtRunCompiledModel failed with LiteRtStatus=3`;
  with real-valued input, runs and returns **wrong numbers** (dev 6.2e-1 against an output
  spread of 4.1e-1)

This is the same defect as bug 2 seen from the other side, and it is why the two should be
investigated together rather than separately. It also blocks the migration path the library
recommends: `GpuDelegate` is already `@Deprecated` pointing at CompiledModel, but
CompiledModel cannot correctly run a shipping model.

Worth fixing on its own merits: where it works, `cm:cpu` is the fastest CPU backend
measured (rtmpose 5.62ms vs XNNPACK 7.74ms).

## Bugs 2 and 4: root cause is upstream LiteRT, not this wrapper

Bugs 2 and 4 are one defect in **LiteRT's own CompiledModel runtime**. They are not
reachable from anything this package does, and cannot be fixed here.

### How that was established

Google publishes its own bindings over the same runtime in the `ai-edge-litert` wheel, which
ships *both* APIs against *one* build of LiteRT: the classic TFLite `Interpreter`
(`ai_edge_litert.interpreter`) and LiteRT Next `CompiledModel`
(`_pywrap_litert_compiled_model_wrapper`). That makes a controlled comparison possible with
this package removed from the picture entirely: same library build, same machine, same model
files, same inputs, only the API differs.

`ai-edge-litert` 2.1.6, macOS arm64, our shipping models, `hardware_accel=CPU`:

| model | Interpreter (6 runs) | CompiledModel |
|---|---|---|
| `cat_face_localizer` | 6/6, dev 1.8e-07 (0.00%) | **6/6, dev 1.8e-07 (0.00%)** |
| `cat_face_landmarks_full` | 6/6, drift 0.0, 56.1ms | **1/6, dev 6.4e-01 = 163% of range, then `code=3`** |

That is the exact pair of symptoms reported above, reproduced with no flutter_litert code
involved. **The earlier framing of this as a wrapper bug was wrong** and is retracted; so is
the intermediate conclusion drawn from testing `ai_edge_litert.interpreter` alone, which
exercises the classic TFLite interpreter and therefore could not say anything about
CompiledModel.

### What triggers it

Synthetic single-op models (`tf.keras` -> TFLite, float32), same harness:

| graph | CompiledModel result |
|---|---|
| conv, depthwise conv | correct, 3/3 |
| 1 `Conv2DTranspose`, deconv is the output op | **wrong, 54% of range** |
| 2 or 3 `Conv2DTranspose`, deconv is the output op | **fails to invoke, `code=3`** |
| conv then deconv (deconv is output) | **wrong / NaN** |
| deconv then conv (conv is output) | correct, 3/3 |
| 2 deconvs then a trivial 1x1 conv (conv is output) | correct, 3/3 |

The naive reading of that table is positional: "a `TRANSPOSE_CONV` writing into the caller's
output buffer breaks, and interposing any op fixes it". That is a symptom, not the cause. The
actual rule is:

> **The bug fires when a model *output* tensor ends up `kTfLiteDynamic`.**

`TRANSPOSE_CONV` is merely a common way to get there. Keras emits its output shape as a
runtime computation, so every `TRANSPOSE_CONV` in these graphs takes a shape input produced by
`PACK` rather than a constant:

| graph | `TRANSPOSE_CONV` | runtime-shaped | model output produced by | result |
|---|---|---|---|---|
| `plain_conv` | 0 | 0 | `CONV_2D` | correct |
| `deconv_x1` | 1 | 1 | **`TRANSPOSE_CONV`** | broken |
| `deconv_x2` | 2 | 2 | **`TRANSPOSE_CONV`** | broken |
| `deconv_x2_then_1x1` | 2 | 2 | `CONV_2D` | correct |

Rows 3 and 4 are identical in the middle two columns. Only the last column differs, and it
decides the outcome: interposing a conv does not remove the dynamic tensor, it demotes it to
an *internal* tensor, so no custom allocation is ever installed on it. Position mattered only
because it determined whether the dynamic tensor was a model output.

For an upstream report the trigger should therefore be described as **"dynamic/runtime-shaped
model output"**, not "multi-layer deconv" and not "deconv as the final op".

### The output is never written at all (mechanism, confirmed)

The first run does not compute a wrong answer. **It writes nothing and reports success.**

Prefill every output element with a sentinel `-7.25`, then call `LiteRtRunCompiledModel`
four times on one instance (native C against the shipped `libLiteRt.dylib`, CPU only,
`update_allocation=false`, all 96 outputs):

```text
run=0 status=0 (kLiteRtStatusOk)               sum=-696 min=-7.25 max=-7.25
run=1 status=3 (kLiteRtStatusErrorRuntimeFailure) sum=-696 min=-7.25 max=-7.25
run=2 status=3 (kLiteRtStatusErrorRuntimeFailure) sum=-696 min=-7.25 max=-7.25
run=3 status=3 (kLiteRtStatusErrorRuntimeFailure) sum=-696 min=-7.25 max=-7.25
```

96 x -7.25 = -696, so the buffer is byte-for-byte untouched while run 0 returns
`kLiteRtStatusOk`. For the same input the classic interpreter returns 96 values spanning
0.305476 to 0.611873, sum 46.955475.

The runtime explains itself in the log:

```text
subgraph.cc:2610 (tensor->allocation_type == kTfLiteArenaRw || ... || kTfLiteNonCpu) was not true
subgraph.cc:1017 tensor_at_index->allocation_type != kTfLiteCustom (4 != 6)
litert_compiled_model.cc:162 Failed to allocate tensors
```

In `TfLiteAllocationType`, 4 is `kTfLiteDynamic` and 6 is `kTfLiteCustom` (verified against
`tflite/core/c/common.h`). So the sequence is:

1. Run 1 installs the caller's TensorBuffer as a TFLite **custom allocation** on the output.
2. Preparation resets that output to **`kTfLiteDynamic`**, and LiteRT computes into its own
   dynamic allocation without ever publishing it back to the caller's buffer. Hence the
   surviving sentinel.
3. Run 2 tries to install the custom allocation again. `SetCustomAllocationForTensor`
   refuses, because the tensor is now dynamic (`subgraph.cc:2610`).
4. `AllocateTensors` then finds allocation type 4 where 6 is required and returns
   `kLiteRtStatusErrorRuntimeFailure` (`subgraph.cc:1017`).

**The wrong numbers and the second-call failure are two stages of one bug**, which is why
they had to be investigated together.

> **Retracted: "the signature is memory corruption."** An earlier version of this section
> argued from values that were deterministic per process but varied across processes, and
> sometimes `NaN`, that LiteRT was reading uninitialized or freed memory. The sentinel test
> shows the simpler truth: the output buffer is never written, so what we measured each time
> was whatever the freshly allocated buffer already contained. No corrupted read is involved.
> Consequently the specific deviation magnitudes throughout this document are incidental, and
> the reproducible assertion is Interpreter/CompiledModel *disagreement*, not any figure.

### Upstream knows, but there is nothing to consume yet

Upstream PR [#8667](https://github.com/google-ai-edge/LiteRT/pull/8667), "Fix dynamic tensor
output buffer synchronization in LiteRT CompiledModel", exists. **Do not read it as a fix that
is about to arrive.** As of 2026-07-29 it is:

- **open and unmerged**, 1 commit, opened 2026-07-13 via copybara, untouched since
- **unreviewed**: 0 reviews, 0 comments
- **absent from `main`**: zero occurrences of `kTfLiteDynamic` in `main`'s
  `litert/runtime/compiled_model.cc`, and no post-invoke copy-back of any kind
- **in no release**: latest is v2.1.6 (2026-07-02), which predates the PR, which is exactly
  why the wheel tested above reproduces the bug

Its value to us is corroboration, not remedy: it confirms Google independently reached the
same root cause, so a fresh bug report would be a duplicate. Treat `Interpreter` on the
affected models as the permanent answer, not a stopgap, because an unreviewed copybara export
can sit indefinitely or be superseded internally.

Its diff adds exactly the missing publication step: after invocation it walks the
outputs and, for any tensor with `allocation_type == kTfLiteDynamic`, locks the caller's
TensorBuffer for write and copies the dynamic allocation into it. The added comment names the
same trigger ("if TFLite operators reset an output tensor to dynamic during preparation").

So this is a known upstream defect with a candidate fix, not something to report cold. Two
caveats, both worth stating in any upstream comment:

- The PR addresses step 2 (the missing copy). Reading the diff, it does not obviously reset
  the dynamic/custom allocation bookkeeping, so **step 4 may survive it** and a second
  invocation could still fail. Our two-run reproduction stays valuable as a regression test.
- Neither of us has built the PR, so that is source inspection, not measurement.

### Why graph surgery will not rescue the real model

The dynamic-output rule also explains why appending an op fixes the synthetics but **not** the
shipping model. `cat_face_landmarks_full` does not end in a deconv, it ends in `RESHAPE` after
soft-argmax (`MUL, SUM, PACK, PACK, RESHAPE`), yet it is still broken, because:

- all 4 of its `TRANSPOSE_CONV`s are runtime-shaped, fed by `SHAPE` -> `STRIDED_SLICE` -> `PACK`
- its output signature is `[-1, 96]`

The dynamic allocation propagates through the decode tail to the model output, so there is no
op you can append to make the output static. Retraining or re-exporting the head with static
shapes might, but that is an ML-repo change with its own risk, and PR #8667 landing would make
it unnecessary. The localizer, with no `TRANSPOSE_CONV` and a static output, is correct.

Also do **not** use "recreate the CompiledModel for every call" as a workaround. It dodges the
second-call `code=3` but every first call still returns the unwritten buffer, which converts a
loud failure into a silent one.

### Scope: this explains the landmark corruption, not the GPU corruption

Two of the three corrupted cells reported in bug 2 are **not** covered by the above and
remain unattributed. Under Google's binding, all three `animal_detection` models are correct
3/3 on both `cpu` and `gpu+cpu`:

| model | `TRANSPOSE_CONV` count | CompiledModel `cpu` | CompiledModel `gpu+cpu` |
|---|---|---|---|
| `species_classifier` | 0 | 9.5e-06 (0.0%) | 9.5e-06 (0.0%) |
| `superanimal_rtmpose_s` | 0 | 3.0e-06 (0.0%) | 3.0e-06 (0.0%) |
| `superanimal_ssdlite` | 0 | 1.7e-06 (0.0%) | 1.7e-06 (0.0%) |

**That comparison is inconclusive for GPU, not exculpatory.** The `gpu+cpu` column is
byte-identical to `cpu`, and the reason is measurable: `IsFullyAccelerated()` returns
`false`, and `hardware_accel=gpu` alone fails to construct at all. The macOS Python wheel
ships no GPU accelerator, so it silently ran CPU both times. The `species_cls` (83%) and
`rtmpose` (40%) corruption we measured appears **only when GPU is in the accelerator set**,
so it is a separate defect that this experiment did not reach, and it could still be ours.
Neither model contains a `TRANSPOSE_CONV`, which independently rules out the deconv
mechanism as its cause.

This is also a concrete argument for binding
`LiteRtCompiledModelIsFullyAccelerated` (already on the to-do list, credited to Codex): it is
the CompiledModel-side equivalent of the bug 1 detector, and it is exactly what distinguished
"ran on GPU" from "silently fell back" here.

### Wrapper audit: nothing here can cause it

Independently of the upstream reproduction, `compiled_model_native.dart` was audited for state
that could produce a first-run/second-run asymmetry. Nothing was found:

- Input/output `TensorBuffer`s are built once (`:302`) and held in fields (`:67-79`). That
  reuse is correct, and matches Google's own `run_model.cc`, which creates buffers once and
  calls `Run` repeatedly.
- `run()` rewrites all inputs, dispatches once, reads all outputs (`:398-431`). No first-run
  cache, no state mutation.
- `_withLockedFloats` (`:563-607`) resets `_lockScratch`, marks locked only after a successful
  lock, and unlocks from `finally` on any exception path. Previously suspected; ruled out.
- `_readOutput` (`:667-682`) copies into a fresh `Float32List`; no runtime-backed view
  survives the call.
- `_dispatchInFlight` (`:537-560`) is async-only and cleared in `finally`.
- Buffer requirements are queried once (`:1117-1197`), which the C contract permits since they
  are CompiledModel-owned and only invalidated by resizing.
- The ranked tensor/layout ABI and the `LiteRtRunCompiledModel` signature match the C headers
  (`litert_ffi.dart:27`, `:295`).

One genuine discrepancy exists and is **not** the cause: the wrapper hardcodes
`update_allocation=true` at `compiled_model_native.dart:326-334`, whereas the official helper
reserves it for dynamic dimensions. Both settings reproduce the bug identically (in Dart and in
the native harness), and flipping it to `0` in a live test made the output far worse
(9.9e+28), so it was reverted. Worth tidying for correctness of intent, not as a fix.

### Consequence

- **No shipping package is currently exposed.** `cat_detection` and `dog_detection` contain
  zero references to `CompiledModel` in `lib/`, so the two affected models are never run on
  the broken path. `animal_detection` gates it behind `useCompiledModel`, which defaults to
  `false`, and none of its three models contain a `TRANSPOSE_CONV`. This is a
  blocked-migration problem, not a live-correctness one.
- **Keep `Interpreter` for the landmark stage.** It is correct and repeatable on both
  platforms and is already the default; nothing in the consuming packages needs to change.
- **Do not migrate the landmark stage to CompiledModel**, despite `GpuDelegate` being
  `@Deprecated` in favour of it. That deprecation is premature for models with a deconv head.
- **This is an upstream matter, not a local fix.** Track PR
  [#8667](https://github.com/google-ai-edge/LiteRT/pull/8667) rather than filing fresh. When
  commenting, ask specifically whether it also reconciles the dynamic/custom allocation state
  before the *next* invocation, since the diff appears to fix the missing copy but not the
  second-call rejection.
- **Bind `LiteRtCompiledModelIsFullyAccelerated`** with optional symbol lookup.
  `LiteRtGetStatusString` needs the same treatment: verified present in the shipped macOS
  dylib, **absent from the shipped `linux/lib/libLiteRt.so`**, so binding it unconditionally
  would break Linux. Fall back to a static enum table there.

### Reproducing

Two harnesses, both CPU-only, both independent of Flutter:

- `doc/repro_compiled_model_transpose_conv.py` needs only
  `pip install ai-edge-litert==2.1.6 tensorflow numpy`. Builds four synthetic graphs and exits
  non-zero when CompiledModel disagrees with the Interpreter. This is the one to attach
  upstream: no proprietary model, no local dylib.
- `build/codex-tmp/repro_native.c` (untracked) drives the shipped dylib directly in C, with the
  sentinel prefill that proves the output is never written. Build:

  ```sh
  clang -std=c11 -Wall -Wextra build/codex-tmp/repro_native.c \
    macos/flutter_litert/Sources/flutter_litert/Resources/libLiteRt.dylib \
    -Wl,-rpath,$PWD/macos/flutter_litert/Sources/flutter_litert/Resources \
    -o build/codex-tmp/repro_native
  build/codex-tmp/repro_native <path-to>/cat_face_landmarks_full.tflite managed
  ```

  Reproduces with both `managed` and host buffers. Model SHA-256
  `b44c94955e9f92ef210bbfdfada7247a3c19c766c66b97d2be82cb0c197d917c`; LiteRT source revision
  `1adc2475829fbe52d5670873821a45bea8779532`; input `input[i] = (i % 251) / 251.0f`.

## Spike result (RESOLVED)

`TfLiteInterpreterModifyGraphWithDelegate` **does** report zero-delegation. Mechanism (a)
is viable: cheap, synchronous, no extra inference.

Called directly on a delegate-free interpreter, its `TfLiteStatus` matches the numerical
ground truth in **20 of 20 cells** (5 models x 2 delegates x 2 platforms), with macOS and
the iOS simulator returning byte-identical results:

| model | `gpu` status | numerically engaged? | `coreml` status | numerically engaged? |
|---|---|---|---|---|
| ssdlite | **0** | yes | 2 | no |
| species_cls | 2 | no | 2 | no |
| rtmpose_s | **0** | yes | 2 | no |
| localizer_b2_224 | 3 | no | 2 | no |
| landmarks_v3l_384 | 3 | no | 2 | no |

Rule: `status == kTfLiteOk (0)` iff the delegate took ops. `kTfLiteDelegateError (2)` and
`kTfLiteApplicationError (3)` both mean it did not.

**What the rule actually guarantees.** Stated precisely, it is one-directional: *non-zero
implies zero ops taken*. The converse is weaker than "engaged". A delegate that takes 3 of
200 ops also returns 0, and the resulting partition boundaries can make it slower than not
delegating at all. Every failing model measured here is a whole-graph refusal, so this does
not affect any result in this document, but it bounds what the fix can promise: it detects
**zero-op refusal**, not degree of delegation.

Nothing better is reachable. The public `TfLiteInterpreter*` surface has no node-count or
execution-plan accessor; `GetExecutionPlan` exists only as a `TfLiteContext` member
(`tensorflow_lite_bindings_generated.dart:4588`), callable from inside a delegate
implementation but not from a `TfLiteInterpreter`. The one adjacent hook,
`TfLiteInterpreterOptionsSetTelemetryProfiler` (`:3348`), is bound but unusable as-is:
`TfLiteTelemetryProfilerStruct` is generated as `ffi.Opaque` (`:5574`), so there is no
constructible layout without hand-writing the struct. The other route to partition counts is
`TfLiteInterpreterOptionsSetErrorReporter` (also bound, also unused), which is how the GPU
delegate reports "N operations will run on the GPU, and M on the CPU"; its callback takes a
C `va_list`, which Dart FFI cannot portably consume on arm64. Treat either as a spike, not a
plan item.

Also unused and worth one line while in this area:
`TfLiteInterpreterOptionsSetEnableDelegateFallback`, TFLite's own mechanism for falling back
to CPU when a delegate fails at *invoke* time. It does not address zero-op, but it is the
same failure family.

Enum values confirmed from `tensorflow_lite_bindings_generated.dart:4146-4162`:
`kTfLiteOk=0`, `kTfLiteError=1`, `kTfLiteDelegateError=2`, `kTfLiteApplicationError=3`.

Two caveats:

- The binding's own docstring claims it "returns one of the following three status codes"
  and lists only Ok / DelegateError / Error. We observe **3 (ApplicationError)** in
  practice, so that documentation is incomplete. Treat any non-zero as "not engaged"
  rather than switching on the specific code.
- It is marked `WARNING: This is an experimental API and subject to change`, and the header
  advises using `TfLiteInterpreterOptionsAddDelegate` "unless absolutely required". Detecting
  zero-delegation is a legitimate instance of "absolutely required", but this is a real
  coupling to an experimental C API and should be guarded accordingly.

### CRITICAL CONSTRAINT: the rule is valid for GPU, NOT CoreML

`ModifyGraphWithDelegate` is only a valid engagement probe for the **GPU/Metal** delegate.
For CoreML it is actively wrong, and using it would disable CoreML entirely.

A/B on MediaPipe models, comparing create-time application (`options.addDelegate`) against
post-creation application:

| model | delegate | create-time | post-creation | agree? |
|---|---|---|---|---|
| face_detection_short_range | **coreml** | dev 2.2e-1 **engaged** | status=2, dev **0.0** | **NO** |
| face_detection_short_range | gpu | dev 1.3e-4 engaged | status=0, dev 1.3e-4 | yes |
| face_landmark | **coreml** | dev 6.6e-2 **engaged** | status=2, dev **0.0** | **NO** |
| face_landmark | gpu | dev 6.9e-5 engaged | status=0, dev 6.9e-5 | yes |

CoreML **cannot be applied after interpreter creation**. `ModifyGraphWithDelegate` does not
merely misreport it: the post-creation deviation is 0.0, i.e. the delegate genuinely did not
attach. Routing all delegates through `ModifyGraphWithDelegate` would silently turn off
CoreML on every model where it currently accelerates.

Evidence tally for the status rule:

- **GPU**: zero disagreements, covering both directions. (Cell count was originally given as
  "22+", which is not reconstructable from the list below. Our 5 models contribute 10 GPU
  cells across 2 platforms, 4 positive and 6 negative; adding the 10 MediaPipe models gives
  **20** if they were probed on one platform and **30** if on both. Record which.)
  - *Positives* (numerically engaged, status 0): all 10 MediaPipe models that could be
    probed (face_blendshapes, face_detection_back/front/full_range/full_range_sparse/
    short_range, face_landmark, iris_landmark, mobilefacenet, selfie_multiclass), plus our
    ssdlite and rtmpose_s.
  - *Negatives* (numerically no-op, status 2 or 3): species_cls, localizer_b2_224 and
    landmarks_v3l_384, on both platforms.
  - Two models (selfie_segmenter, selfie_segmenter_landscape) could not be probed by the
    harness ("failed precondition"); this is a harness limitation, not a rule failure.
- **XNNPACK**: applies cleanly post-creation, status=0, output bit-identical to the
  existing options path.
- **CoreML**: agrees only where CoreML does nothing anyway; disagrees on every model where
  it works. Rule invalid.

So the fix must be **delegate-specific**: probe GPU with `ModifyGraphWithDelegate`, keep
CoreML on the create-time path, and use the numerical check if CoreML engagement ever needs
verifying.

### Verified: the fix flow works end-to-end

Fresh interpreter -> apply GPU -> non-zero status -> apply XNNPACK to the **same**
interpreter -> run:

| model | gpu status | xnnpack status | result dev | xnnpack-via-options dev |
|---|---|---|---|---|
| landmarks_v3l_384 | 3 | **0** | 3.1e-6 | 3.1e-6 (identical) |
| localizer_b2_224 | 3 | **0** | 6.0e-8 | 6.0e-8 (identical) |
| ssdlite | **0** (keeps GPU) | n/a | 1.9e-6 | n/a |

The fallback reproduces today's XNNPACK path exactly on a single-delegate interpreter.

> **Retracted by the merged review.** This was originally stated as "no interpreter rebuild
> is required". That does not generalise: `kTfLiteDelegateError` undoes *all* previously
> applied delegates, so reusing the interpreter can silently drop an earlier delegate (e.g.
> Flex), and status 3 carries no documented restoration guarantee at all. Two passing
> single-delegate experiments are not sufficient. **Rebuild the interpreter after a failed
> apply.**

### Verified: safety after a failed apply

- **The interpreter stays usable and correct.** After `status=3`, `landmarks` runs with
  dev 0.0 against a clean CPU reference. A partially-modified graph would have shown drift.
- **The rejected delegate appeared safe to free immediately** in testing: deleting the GPU
  delegate *before* running, after a failed apply, survived with finite outputs.
  **Do not rely on this.** The C contract grants no such exception, and status 3 has no
  documented restoration guarantee, so one passing observation is not a licence. Retain
  every attempted delegate until the interpreter is destroyed.

### Verified: the probe is cheap enough to be always-on

Cost of `GpuDelegate()` construction plus `ModifyGraphWithDelegate`, macOS:

| model | probe | outcome |
|---|---|---|
| landmarks_v3l_384 | **0.1 ms** | rejects |
| localizer_b2_224 | **0.2 ms** | rejects |
| species_cls | **0.5 ms** | rejects |
| ssdlite | 7.5 ms | accepts |
| rtmpose_s | 16.7 ms | accepts |

Rejection costs 0.1-0.5 ms, i.e. free. The 7.5-16.7 ms on acceptance is the delegate
compiling the graph, which the current `options.addDelegate` path already pays inside
`TfLiteInterpreterCreate`; the probe relocates that cost rather than adding it.

So the failure case we are fixing costs well under a millisecond to detect. There is no
need to make this opt-in or debug-only.

### Compounding harm: no-op delegates also disable IsolateInterpreter

`InterpreterFactory.createIsolateIfNeeded` returns null when
`interpreter.hasActiveDelegate` is true. Because a zero-op delegate still sets that flag,
today's silent failure costs **twice**: no acceleration, *and* no `IsolateInterpreter`, so
inference stays on the calling thread on every platform except macOS (which opts out
separately).

Falling back to XNNPACK does not restore the isolate path, since XNNPACK is itself a
delegate and sets the same flag. That is the correct trade (acceleration beats isolation
here), but `hasActiveDelegate` conflates "a delegate is attached" with "a delegate is
doing work", and should be split once engagement is known.

### Consequence for the design

- **Bug 1** can be fixed with always-on production behavior: create the interpreter, apply
  the delegate via `ModifyGraphWithDelegate`, check the status, and fall back to XNNPACK on
  non-zero. No numerical check, no extra inference.
- **Bug 2 still needs the numerical self-check.** CompiledModel is a different API, and its
  failure is wrong output from a call that *succeeds*. `landmarks` + `cm:cpu` returns
  status-clean garbage. No status code will catch that.

So both mechanisms are needed, but for different bugs, and only the cheap one is on the hot
path.

## Impact on the seven consumer packages

`face_detection_tflite`, `pose_detection`, `hand_detection`, `cat_detection`,
`dog_detection`, `animal_detection`, `object_detection` all pin `flutter_litert: ^3.6.0`.

> **Qualified by the merged review.** Two caveats found later: (1) four sites across three
> packages delete the delegate *before* closing the interpreter, violating the C contract
> that "'delegate' must outlive the interpreter"
> (`tensorflow_lite_bindings_generated.dart:3071`). Two of them are shared dispose mixins,
> so the affected class count is 8+, not 4. Full list in step 4 of the suggested order. That
> is a pre-existing use-after-free ordering bug, unrelated to this fix, and should be
> corrected in those packages. (2) If CompiledModel's packed-size/stride handling is
> corrected, downstream code that treats allocation byte sizes as logical shapes (e.g.
> `object_detection/lib/src/models/object_detection_model.dart:204`) may need updating.

**For the Interpreter fallback specifically, they need no code changes.** Fixing it means all
seven get correct behavior from a version bump alone. That is the argument for fixing this
here rather than in each package.

Optional additions, in increasing cost:

1. **Nothing.** The fallback fix is the whole benefit. Seven pubspec edits.
2. **A process-wide diagnostics hook** for apps wanting telemetry:
   ```dart
   LitertDiagnostics.onDelegateEvent = (e) { /* requested, effective, engaged */ };
   ```
   Registered once by the **app**, never by the packages. No changes in any of the seven.
3. **A query property** for tests:
   ```dart
   interpreter.acceleration.engaged   // "not refused", NOT "fully delegated"
   interpreter.acceleration.effective
   ```
   Name and document `engaged` against what the probe can actually establish (see *What the
   rule actually guarantees* above): it means the delegate did not refuse the graph, not that
   it took every op. `notRefused` is uglier and more honest; either way the doc comment must
   not promise full delegation.
   Their integration tests already depend on `flutter_litert` transitively and can import
   it directly, so this needs no re-export either. Worth having: it turns this
   investigation into a regression test, so a future model export that silently stops being
   delegable gets caught.

Do **not** add an `accelerationInfo` getter to all seven detectors. Seven API surfaces and
seven publishes to expose something the app can already read from the library.

## Suggested order

Revised after the merged review.

**Ship independently, no dependencies:**

1. Bind `LiteRtGetStatusString` **behind a nullable lookup**, and keep the bug 3 integer map
   as the fallback. It is not available on every platform:

   | shipped binary | `LiteRtGetStatusString` | `LiteRtCompiledModelIsFullyAccelerated` |
   |---|---|---|
   | macOS `libLiteRt.dylib` | yes | yes |
   | iOS `libLiteRt.dylib` | yes | yes |
   | Android arm64 `libLiteRt.so` | yes | yes |
   | `linux/lib/libLiteRt.so` | **no** | yes |
   | `windows/libLiteRt.dll` | **no** | yes |

   Verified with `nm` for Mach-O and `objdump -T` for ELF. Linux exports 402 `LiteRt*`
   symbols and this is not one of them.

   This is not a cosmetic concern. `LiteRtBindings` resolves every symbol **eagerly in its
   constructor initializer list** (`lib/src/bindings/litert_ffi.dart:97` onward), and the
   file has no `tryLookup` pattern to copy. Adding the field naively throws at
   bindings-construction time and takes down *all* CompiledModel usage on Linux and Windows,
   both of which run `compiled_model_test.dart` in CI (`.github/workflows/flutter-ci.yml:161`
   and `:212`). The guard is required, not optional.
2. Bind `LiteRtCompiledModelIsFullyAccelerated` for CompiledModel acceleration reporting.
   Present on all five binaries, so this one needs no guard.
3. Fix the two confirmed leaks: `CoreMlDelegate()` options (`coreml_delegate_native.dart:68`)
   and `InterpreterPool.initialize` options (`interpreter_pool.dart:80`).
4. Fix delegate/interpreter teardown order in the consumer packages. This is **four code
   sites, not two**, and two of them are shared dispose mixins rather than leaf call sites:

   | package | site | note |
   |---|---|---|
   | `object_detection` | `lib/src/util/helpers.dart` `_doDispose` | mixin, used by `ObjectDetection` |
   | `face_detection_tflite` | `lib/src/util/helpers.dart:73` `_doDispose` | mixin, used by `FaceBlendshapesModel`, `FaceEmbedding`, `FaceLandmark`, `SelfieSegmentation`, `IrisLandmark` |
   | `face_detection_tflite` | `lib/src/models/face_detection_model.dart:525` | hand-rolled copy of the same sequence |
   | `pose_detection` | `lib/src/models/person_detector_native.dart:191` | deletes the delegate, then `disposeBase()` closes the interpreter at `person_detector_base.dart:32` |

   `animal_detection` (`single_interpreter_model.dart:144`) already has the right order;
   `hand_detection`, `cat_detection` and `dog_detection` never delete a delegate.

   That the sync and async paths of the *same* `face_detection_tflite` mixin disagree
   (`_doDisposeAsync` at `helpers.dart:83-85` closes the interpreter first, correctly)
   confirms this is an oversight rather than a deliberate ordering.

**Bug 1, verified GPU fallback:**

5. Build the construct-apply-allocate path. Not in `InterpreterFactory.create`, which
   cannot apply a delegate; this needs a new internal sequence that owns the interpreter.
6. Apply to **GPU only**. CoreML must stay on the create-time options path.
7. **Rebuild** the interpreter after a failed apply rather than reusing it, since
   `kTfLiteDelegateError` undoes all previously applied delegates.
8. Resolve delegate ownership explicitly: who holds the rejected delegate, the effective
   delegate, and what the returned `Delegate?` now means for the pool.
9. Test on physical iOS and physical Android (GPUv2) before shipping. Simulator and macOS
   share host hardware and are not independent observations.

**Bug 2, CompiledModel correctness:**

10. First correct packed-size / stride / alignment handling, which is a real defect even
    though it does not explain the `landmarks` corruption (that model has no padding).
11. Re-run the matrix. If corruption persists, build a native C/C++ parity runner against
    the pinned LiteRT API to establish whether the fault is in LiteRT or in this wrapper.
12. Only then consider a verifier, and prefer opt-in or caller-supplied calibration vectors
    over an unconditional constructor-time numerical check.

**Then:**

13. Extend the existing `full_delegate_sweep_test.dart` with numerical (not just load)
    verification, rather than adding a parallel harness.
14. Bump the seven packages.

Steps 1-4 are behavior-preserving fixes. Step 5-9 is a behavior improvement with no API
break: **3.7.0**. Step 10-12 may force a semantic change to `inputByteSizes`/
`outputByteSizes` and could require a major.

## Reproducing

The matrix harness is not committed. It loads each model, runs every backend, and records
latency plus max deviation from the `interp:disabled` reference, catching per-cell failures
so one broken backend does not abort the run. Output format:

```
MTX|<platform>|<model>|<backend>|<ms>|<dev>|<status>
```

Rebuild it as an integration test under any package that bundles models, then:

```
flutter test integration_test/matrix_test.dart -d macos
flutter test integration_test/matrix_test.dart -d <simulator-id>
```

Worth committing as a regression test once the fixes land.

## Caveats

- All GPU measurements are macOS and iOS **simulator**. The simulator's Metal runs on the
  host GPU through a translation layer. The **no-op findings are op-support facts** and
  hold everywhere; the **relative GPU timings are not** predictive of device.
- Not yet run on physical hardware. `Hugo's iPhone` (iOS 26.5) is visible to
  `xcrun xctrace list devices` but was not connected during this investigation.
- The macOS-vs-iOS CPU gap (1.3-1.9x generally, 3.05x for `landmarks_v3l_384`) is real and
  reproducible but not diagnosed. Separate issue; desktop-only, does not affect shipped
  mobile apps.

---

# Merged review: independent adversarial analysis

A second analyst (Codex) reviewed this document with instructions to attack it. Its findings
are merged below. Everything it raised was independently verified before being accepted;
disputed items are recorded with the disproof and excluded from recommendations.

## Confirmed by both, independently

- `fromBufferWithGpuFallback` cannot detect a successful `run()` returning bad values.
  Codex adds a precision: it catches errors during compilation **and buffer/signature
  setup**, not compilation alone. It cannot catch `run()` errors because no inference
  happens during construction.
- Android is entirely untested; physical iOS is untested.
- XNNPACK could itself silently accept zero nodes, so falling back to it is not a guarantee
  of acceleration.

## Accepted from Codex (verified)

| # | Finding | Verification |
|---|---|---|
| 1 | **`status == 0` is not a contract guarantee.** TFLite ignores an empty replacement set and returns `kTfLiteOk`; NNAPI explicitly returns success with zero nodes delegated. | Cannot verify upstream source from here, but this is a contract-level claim and our data cannot refute it: we only show that *Metal/GPU* happens to return 2 or 3. The rule is **empirically validated for GPU, not contract-guaranteed**. Materially weakens the "always-on, all delegates" framing. |
| 2 | **macOS and iOS-simulator are not independent observations.** They share host hardware. | Correct. "20/20 across 2 platforms" is really ~1.5 platforms. Sample breadth comes from the 10 MediaPipe models, not from platform diversity. |
| 3 | **`InterpreterFactory.create` is the wrong layer for the fix.** It receives no model or interpreter and returns only `(InterpreterOptions, Delegate?)`. | Verified at `interpreter_factory.dart:29`. The fix needs an internal construct-apply-allocate sequence, not a change to `create`. This invalidates the fix's stated location. |
| 4 | **`CoreMlDelegate()` leaks its options** when constructed without them. | Verified, `coreml_delegate_native.dart:68-69`: `options ?? CoreMlDelegateOptions()` is never deleted. Only affects direct construction; `_createCoreml` passes and deletes its own. |
| 5 | **`InterpreterPool.initialize` leaks `InterpreterOptions`** once per slot. | Verified, `interpreter_pool.dart:80-82`: the delegate is retained in `_delegates`, the options object is not deleted. |
| 6 | **Web accepts delegates and ignores them while recording `hasDelegate`.** | Verified, `web/interpreter_options.dart:1-12`. The same silent-no-op class, by design. Any engagement API needs a distinct web implementation. |
| 7 | **`LiteRtCompiledModelIsFullyAccelerated` exists and is not bound.** | Verified: no match in `lib/` or `src/`. Directly relevant, and better than inferring acceleration. Confirmed exported by all five shipped `libLiteRt` binaries, so it can be bound unconditionally. |
| 8 | **`LiteRtGetStatusString` exists and is not bound.** | Verified not bound. **But the "supersedes the bug 3 map" conclusion is wrong:** the symbol is absent from the Linux and Windows binaries this package ships. See step 1 of the suggested order. Both are needed. |
| 9 | **The repo already has 14 integration tests**, including `full_delegate_sweep_test.dart`, whose header states delegates are built directly "so a delegate that fails to load throws here instead of silently becoming CPU". | Verified. A genuine miss: existing coverage should have been surveyed before building a new harness. That sweep catches *load* failures; it does not catch the zero-op case, so the numerical check still adds something. |
| 10 | **`kTfLiteDelegateError` undoes all previously applied delegates**, so a GPU failure can silently remove an earlier delegate (e.g. Flex). Status 3 has no documented restoration guarantee at all. | Header text confirmed. Two successful same-interpreter experiments do not generalise. **Retracts this document's "no interpreter rebuild is required" claim** for the multi-delegate case. |

## Disputed (tested and disproved, excluded from recommendations)

| # | Codex claim | Disproof |
|---|---|---|
| A | The CoreML result is **allocation-confounded**: `Interpreter._` auto-allocates (`interpreter.dart:82-84`), so the A/B compared pre- vs post-allocation, not create-time vs post-creation. | The confound is **real and well spotted**, but the conclusion survives. Rebuilt the test natively (`TfLiteInterpreterCreate` -> `ModifyGraphWithDelegate` -> `AllocateTensors`), applying the delegate **before** allocation: CoreML still returns **status=2** on both MediaPipe models where it engages via options. GPU still returns 0 / 3 as before. So allocation order is not the variable; the application *mechanism* is. The claim is retained with a corrected rationale. |
| B | RTMPose's deviation may be an **output-ordering** artifact; it has two same-shaped `[1,39,512]` outputs. | Ran a best-match permutation search: every CM output matches its **own** index best. `cm:cpu` is index-aligned at 3.3e-6 / 3.9e-6 (correct), `cm:gpu+cpu` at 2.7e-1 / 2.5e-1 with no better permutation. Ordering does not explain it. |
| C | The ssdlite "control validates the harness" claim is invalid if only one of twelve outputs was checked. | Applies to the repo's committed `engine_matrix_test.dart:214`, not to this harness, which concatenates all outputs. Re-verified per-output: all 12 individually within 7.7e-7 to 1.0e-5. Control stands. |
| D | **Tensor-buffer packing** is the largest missed confounder: the wrapper uses `LiteRtGetTensorBufferRequirementsBufferSize` (backing allocation) as the logical size, ignoring packed size, strides and alignment. | A serious and plausible mechanism in general, and worth fixing on its own merits. But it **cannot explain `landmarks`**: input `[1,384,384,3]` = 442,368 floats = 1,769,472 bytes and `inputByteSizes` reports exactly 1,769,472; output `[1,96]` = 384 bytes and `outputByteSizes` reports exactly 384. No padding exists for this model, yet it is corrupt. Retained as a separate defect to investigate, not as the explanation for bug 2. |

## Net effect on recommendations

Codex's strongest contributions are #1, #3, #9 and #10. Together they mean:

- The fix cannot be described as "always-on for all delegates" (#1) and cannot live in
  `InterpreterFactory.create` (#3).
- The "no rebuild required" shortcut is not safe in the presence of other delegates (#10),
  so the implementation should rebuild rather than reuse after a failed apply.
- Existing test coverage must be extended rather than replaced (#9).

Its CompiledModel critique (#D) did not explain the observed corruption but did surface a
real packing/stride defect worth fixing independently.

Revised shipping position: the **GPU-only, verified-fallback** change remains justified, but
should rebuild the interpreter on failure rather than reuse it, must not be applied to
CoreML, and needs physical-device coverage on iOS and Android before shipping. Binding
`LiteRtGetStatusString` and `LiteRtCompiledModelIsFullyAccelerated` can ship independently
and immediately.
