# Authoritative macOS model-matrix results

Run completed on 2026-08-03 with a complete 435-cell dataset: 29 published
models multiplied by 15 Interpreter/CompiledModel configurations. The raw
results are in [JSON](MACOS_MODEL_MATRIX_RESULTS.json) and
[CSV](MACOS_MODEL_MATRIX_RESULTS.csv).

This supersedes the 2026-08-01 run recorded at commit `a6d0297`. Every mode
reproduced its earlier counts exactly except Interpreter Core ML, which moved
from 20 executing / 12 accurate to 24 / 13. That is the patched Core ML dylib
from `c8e0386` landing: four models whose global-spatial `MEAN` was previously
rejected now compile, and one of the four is accurate. The reproduction of the
other fourteen modes is itself evidence that generalizing the harness to cover
iOS changed no macOS behaviour.

## Run and dataset integrity

- Host: Mac16,5, arm64, macOS 26.4 (25E246), 16 logical processors
- Build: Flutter profile; `flutter_litert` commit `c6e4d4b`
- Interpreter runtime: `2.20.0-dev0+selfbuilt`
- Timing: 5 warmups followed by 15 measured samples per successful path
- Accuracy: three finite deterministic fixtures per model, compared with an
  independent plain-CPU Interpreter reference using `1e-4 + 1%` scaled
  tolerance
- Inventory: 29/29 model references succeeded and every reference output was
  finite
- Shape: 435 expected rows, 435 actual rows, 435 unique rows; every mode has 29
  rows and every model has 15 rows
- CSV: 435 data records, 80 columns, with consistent record width. This run's
  per-cell `row_started_utc` is present in the JSON rows but not the CSV: the
  driver and the host runner keep separate column lists and only the driver's
  was updated in time. The host list is fixed, so later runs carry it in both.
- Orchestration: 21 process attempts, including automatic isolation of three
  native terminations; no model row was lost

Overall status was 269 `ok`, 62 recoverable `error`, 101 `unsupported`, one
`native_crash`, and two `native_termination`. Of the 269 configurations that
executed, 216 passed tensor parity and 53 failed it. The quality gate is
therefore false by design: the dataset is complete, but it found real backend
compatibility and accuracy failures.

## Interpreter

`Accurate` and `inaccurate` partition the `ok` column. Warm p50 and setup are
medians of each successful model's measured value, so they summarize models of
very different sizes and should not replace the per-model rows. Interpreter
warm timing is invoke-only.

| Mode | OK | Accurate | Inaccurate | Error | Unsupported | Warm p50 (ms) | Setup median (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU, 4 threads | 29 | 29 | 0 | 0 | 0 | 3.618 | 0.358 |
| XNNPACK, 4 threads | 29 | 29 | 0 | 0 | 0 | 2.764 | 1.963 |
| Metal fp16-capable | 24 | 11 | 13 | 0 | 5 | 0.765 | 16.586 |
| Metal fp32 | 24 | 23 | 1 | 0 | 5 | 1.002 | 15.730 |
| Core ML, all devices | 24 | 13 | 11 | 1 | 4 | 3.602 | 357.106 |
| GPUv2 GL/CL API | 0 | 0 | 0 | 0 | 29 | - | - |
| Flex / Select TF Ops | 29 | 29 | 0 | 0 | 0 | 7.898 | 0.516 |

The plain CPU, XNNPACK, and Flex paths passed all models. Across the 28
accuracy-valid models with non-zero measurable CPU and XNNPACK p50 values,
XNNPACK was faster for 22 and had a median 1.14x speedup.

Metal fp32 is the strongest general Interpreter accelerator in this run: 23 of
24 executing models passed parity, and it was faster than CPU for 18 of 22
measurable accuracy-valid comparisons, with a median 3.92x speedup. The one
executing accuracy failure was
`animal_detection/superanimal_rtmpose_s_float16`; five other models were
strictly rejected rather than silently measured after CPU fallback. Metal
fp16-capable execution was faster for many models but failed parity for 13 of
24, so it is not a safe default.

Core ML had a high median setup cost, four strict delegate rejections, one
recoverable invocation error, and eleven parity failures among its 24 executing
models. The patched dylib moved four models from rejected to executing, but
only one of those four is accurate, so "compiles" still does not imply "safe to
select" and Core ML remains unsuitable as an automatic choice. Flex was genuinely bundled and all 29 models ran, but it is a
compatibility delegate rather than an accelerator. The explicit partition log
available for `face_blendshapes` reported 0 of 182 nodes delegated to Flex,
which is expected for a builtin-only model. GPUv2 is not implemented for macOS
and all 29 rows report that restriction at delegate creation.

## CompiledModel

Every successful CompiledModel row exercised both `run` and `runAsync`, checked
both paths for parity, and collected 15 warmed samples for each. Its timing
scope includes managed I/O, so values should be compared within CompiledModel,
not directly with Interpreter invoke-only values.

| Accelerators | OK | Accurate | Inaccurate | Error | Unsupported | Native | Sync p50 (ms) | Async p50 (ms) | Compile median (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU fp32 | 27 | 27 | 0 | 2 | 0 | 0 | 2.766 | 2.728 | 2.789 |
| GPU fp16 | 18 | 4 | 14 | 10 | 0 | 1 | 1.096 | 1.117 | 12.137 |
| GPU fp32 | 18 | 18 | 0 | 10 | 0 | 1 | 1.321 | 1.190 | 12.870 |
| NPU fp32 | 1 | 1 | 0 | 28 | 0 | 0 | 0.025 | 0.037 | 5.014 |
| GPU + CPU fp32 | 22 | 20 | 2 | 6 | 0 | 1 | 1.537 | 1.405 | 14.895 |
| NPU + CPU fp32 | 24 | 12 | 12 | 5 | 0 | 0 | 4.592 | 4.655 | 398.750 |
| NPU + GPU fp32 | 0 | 0 | 0 | 0 | 29 | 0 | - | - | - |
| NPU + GPU + CPU fp32 | 0 | 0 | 0 | 0 | 29 | 0 | - | - | - |

Compiled CPU was accurate for all 27 models it could run. The two runtime
failures were `cat_detection/cat_face_landmarks_full` and
`dog_detection/dog_face_landmarks_full`, both during the first synchronous
accuracy invocation.

GPU fp32 was accurate for all 18 models that compiled. Against Compiled CPU on
the same accuracy-valid models, it was faster for 13 of 18 with a median 2.42x
speedup. GPU + CPU expanded execution to 22 models; 20 were accurate, 15 of
those 20 were faster than Compiled CPU, and the median speedup was 2.78x. Its
two parity failures were `animal_detection/species_classifier_float16` and
`animal_detection/superanimal_rtmpose_s_float16`. GPU fp16 passed only 4 of 18
executing models and should remain opt-in.

Strict NPU placement succeeded only for
`hand_detection/canned_gesture_classifier`, with all 4 nodes in one Core ML
partition. NPU + CPU fallback executed 24 models but passed parity for only 12,
had a 406.7 ms median compile cost, and was faster than Compiled CPU for only 2
of 12 measurable accuracy-valid comparisons. On Apple, NPU cannot be combined
with GPU in the current API; both requested combinations are explicitly and
consistently represented by 29 `unsupported` rows.

For the multi-model modes, warmed async and sync medians were nearly identical:
the median async/sync ratio was between 0.98 and 1.02. Async remains useful for
scheduling, but this run does not show a material single-inference latency
advantage. The lone strict-NPU model is too small for that aggregate comparison.

## Accuracy failures

All 53 failures were finite-output mismatches rather than reference failures.
They were concentrated in approximate or partially delegated execution:

| Mode | Accuracy failures |
|---|---:|
| Interpreter Metal fp16 | 13 |
| Interpreter Metal fp32 | 1 |
| Interpreter Core ML | 11 |
| Compiled GPU fp16 | 14 |
| Compiled GPU + CPU fp32 | 2 |
| Compiled NPU + CPU fp32 | 12 |

The worst tolerance ratios ranged from 1.02x to 163.45x, so the failures are
not merely one uniformly over-strict boundary. The raw rows contain every
fixture/output comparison, maximum absolute and relative errors, top-index
diagnostics, and sync-versus-async results. CPU, XNNPACK, Flex, Compiled CPU,
and every successfully compiled GPU fp32 row had zero parity failures.

This layer tests deterministic backend regression, not task-level quality on
real images. A later labeled-image suite can add IoU/mAP for detectors and
normalized landmark/keypoint metrics without weakening this tensor-parity
gate.

## Error handling and native failures

Recoverable errors are catchable and now preserve the operation, numeric code,
and symbolic status name. All 62 `error` rows have structured status data:

- 31 `kLiteRtStatusErrorCompilation` (504) errors during compile
- 27 `kLiteRtStatusErrorRuntimeFailure` (3) errors during compile
- 3 `kLiteRtStatusErrorRuntimeFailure` (3) errors during synchronous accuracy
  invocation
- 1 Interpreter `kTfLiteError` (1) during accuracy invocation

For example, the Core ML blendshapes failure is recorded as
`TfLiteInterpreterInvoke failed with TfLiteStatus=1 (kTfLiteError)`, together
with its `accuracy` phase, 54/182 delegated nodes in 29 partitions, and the
native Core ML prediction diagnostics. Unsupported rows likewise retain their
exact phase and reason instead of being collapsed into generic failures.

An uncaught native C++ exception or signal cannot be caught by Dart. The
process-isolated runner still exposes and attributes it by logging model/mode
phase markers, preserving the fatal native line, collecting an Apple crash
report when available, and continuing unaffected cells. In all three Compiled
GPU configurations, `face_detection_full_range_sparse` terminated during
compile with:

```text
std::bad_optional_access: bad_optional_access
205 operations will run on the GPU, and the remaining 183 operations will run on the CPU.
```

The GPU-fp16 occurrence has an `EXC_CRASH/SIGABRT` `.ips` report with native
Metal accelerator frames. macOS did not emit another report for the two
immediately repeated occurrences, so those rows are classified as
`native_termination` from the identical fatal C++ evidence rather than as
generic process failures. The JSON embeds the evidence; all three rows retain
their originating shard log paths.

## Initial backend policy suggested by this run

- Interpreter: use XNNPACK as the broad default. Enable Metal fp32 through a
  per-model allowlist after parity checks; do not silently fall back and count
  the result as Metal.
- CompiledModel: keep CPU as the broad baseline. GPU fp32 is the best current
  accelerated candidate for an allowlist of the 18 models that compile and
  pass. GPU + CPU can broaden that allowlist to 20 accurate models.
- Keep GPU fp16 and NPU + CPU experimental until their parity failures are
  understood. Exclude the sparse full-range face detector from all Compiled
  GPU configurations until the native `bad_optional_access` bug is fixed.
- Treat Flex as Select TF Ops compatibility, not as a performance backend.

Reproduce the complete collection without stopping on the expected quality
failures:

```sh
test/benchmark/run_apple_model_matrix.sh --no-enforce
```
