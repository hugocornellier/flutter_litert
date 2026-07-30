# macOS CompiledModel NPU

## Scope

`Accelerator.npu` is implemented for macOS 13+ on Apple Silicon. This document
covers the macOS implementation; the separate iOS path is documented in
[`ios_compiled_model_npu.md`](ios_compiled_model_npu.md).

The implementation consists of:

- `libLiteRtCoreMlNpuAccelerator.dylib`, a small LiteRT accelerator-registration
  bridge;
- `libtensorflowlite_coreml_npu-mac.dylib`, a dedicated Core ML delegate built
  from LiteRT v2.1.5;
- macOS-only lazy registration in the native `CompiledModel` path.

CPU-only and GPU-only models keep using the existing shared LiteRT environment
and do not load the NPU binaries.

## Placement semantics

The dedicated delegate sets:

```objc
configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
```

This excludes Metal/GPU placement. Apple does not provide a
Neural-Engine-only `MLComputeUnits` value, so Core ML may use the CPU for layers
that the ANE cannot execute.

- `{Accelerator.npu}` is strict at the TFLite layer. LiteRT returns
  `kLiteRtStatusErrorCompilation` if any TFLite operation remains outside the
  Core ML delegate.
- `{Accelerator.npu, Accelerator.cpu}` applies Core ML first, then XNNPACK to
  the remainder. A dedicated environment is necessary because an accelerator
  registered after LiteRT's default XNNPACK registration would otherwise see
  an already-delegated graph and claim zero nodes.
- A mixed request that delegates zero Core ML nodes throws instead of silently
  returning CPU-only inference.
- Combining NPU and GPU is rejected on macOS for now. Supporting it correctly
  requires an explicit, tested ordering policy rather than relying on registry
  order.

`CompiledModel.isFullyAccelerated` means all graph nodes were claimed by some
selected delegate. In mixed mode, Core ML plus XNNPACK—or XNNPACK alone in an
unprotected implementation—can therefore make it `true`. It does not identify
the ANE. The zero-node native guard proves Core ML claimed work; a fixed-input
output comparison with `verifyCompiledModel` is still required to validate the
result numerically.

## Delegate fixes and options

The build uses Core ML version 3, delegates all eligible partitions, and keeps
the upstream/default minimum of two nodes per partition.

It also carries the required `MEAN` pooling fix:

```cpp
pooling_params->set_globalpooling(true);
pooling_params->mutable_valid();
```

Without the padding field, Core ML rejects any generated partition containing a
global `MEAN`. TFLite may then restore a CPU plan without surfacing the original
delegate failure. The species-classifier regression test contains ten `MEAN`
operations specifically to catch this.

## Validation matrix

Measured on an Apple M4 Max running macOS 26.4. Each row uses one identical,
deterministic ramp input for a bare-CPU Interpreter and for
`{Accelerator.npu, Accelerator.cpu}`. “Deviation” is the maximum absolute
difference divided by the reference output range. These are correctness checks,
not portable performance claims.

| Model | Core ML nodes | Deviation | Result |
|---|---:|---:|---|
| `simple_model` | 1 / 1 | 0.000% | pass |
| `add` | 2 / 2 | 0.000% | pass |
| `face_detection_short_range` | 73 / 164 | 0.034% | pass |
| `mobilefacenet` | 163 / 231 | 0.282% | pass |
| `species_classifier_float16` | 140 / 280 | 0.484% | pass; includes 10 `MEAN` ops |
| `efficientdet_lite0` | 238 / 263 | 0.302% | pass |
| `yolov8n_float32` | 215 / 252 | 0.518% | pass |
| `selfie_multiclass` | 115 / 175 | 4.160% | rejected by default verification tolerance |
| `pose_landmark_heavy` | 279 / 689 | 7.231% | rejected by default verification tolerance |
| `superanimal_rtmpose_s_float16` | 155 / 290 | 61.692% | rejected; unsafe on this backend |

The three failing rows reproduce when the dedicated Core ML delegate is attached
directly to the classic Interpreter, without the LiteRT accelerator bridge.
They are Core ML delegate/model compatibility results, not bridge or buffer
plumbing failures. Do not enable NPU for those models without application-level
accuracy validation.

Latency is deliberately absent from this table. Timing cannot distinguish an
accelerator from a slow CPU fallback, and macOS performance does not predict
iPhone performance.

## Reproducible build

`.github/workflows/build-coreml-npu-macos.yml` pins LiteRT v2.1.5 at commit
`9d26e89d88ef8785b6a1e54ec41ac8add215a125`. It applies
`patches/litert_coreml_npu_macos.patch`, builds the arm64 delegate and bridge,
sets stable install names, signs both artifacts ad hoc, verifies their exported
symbols, and uploads the pair.

The packaged files require macOS 13 and arm64. The rest of the plugin retains
its existing macOS deployment target because these dylibs are loaded only after
an explicit NPU request.
