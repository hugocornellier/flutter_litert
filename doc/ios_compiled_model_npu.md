# iOS CompiledModel NPU

## Status

Shipped and validated on a physical device. Both distribution channels carry
the patched Core ML framework with the NPU entry points:

- SwiftPM pins `TensorFlowLiteCCoreML` to the `coreml-ios-v1.1.0` release;
- CocoaPods downloads it from `libs-v0.1.9`, and the podspec re-downloads when
  the NPU symbol is missing rather than trusting a cached file.

Earlier releases pointed both channels at a framework predating the NPU entry
points, so accelerator registration failed on device with
`kLiteRtStatusErrorUnsupported`. See the 3.8.0 entry in
[CHANGELOG.md](../CHANGELOG.md).

Measured on a physical iPhone 15 Pro (iOS 26.5, 2026-08-05), iOS matches macOS
exactly: strict `{npu}` places a full graph for 1 of 29 published models, and
`{npu, cpu}` runs 24 of 29 with 12 matching a bare-CPU reference. Raw results
are in
[IOS_MODEL_MATRIX_RESULTS.json](../test/benchmark/IOS_MODEL_MATRIX_RESULTS.json).

## Placement semantics

The patched Core ML framework keeps the classic Interpreter delegate on
`MLComputeUnitsAll` and adds a separate CompiledModel entry point configured
with:

```objc
configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
```

Apple does not expose a Neural-Engine-only compute-unit mode. This policy
excludes the GPU while allowing Core ML itself to schedule between the Neural
Engine and CPU.

- `{Accelerator.npu}` requires the entire TFLite graph to be accepted by the
  Core ML delegate.
- `{Accelerator.npu, Accelerator.cpu}` gives Core ML first choice and registers
  XNNPACK afterward for remaining operations.
- A zero-node Core ML result is rejected rather than silently returning
  CPU-only inference.
- NPU and GPU cannot currently be combined on Apple platforms.

The iOS accelerator bridge mirrors the ABI of the bundled LiteRT runtime at
commit `1adc2475829fbe52d5670873821a45bea8779532`. That revision wraps a TFLite
delegate together with its deleter; it is intentionally different from the
newer delegate-lifetime ABI used by the macOS build.

## Simulator validation

The integration suite
`example/integration_test/ios_compiled_model_npu_test.dart` passes on an arm64
iPhone 16 simulator running iOS 18.2:

1. strict NPU compilation, inference, and full-graph ownership for
   `simple_model`;
2. NPU+CPU agreement with a bare-CPU Interpreter across
   `species_classifier_float16`, `mobilefacenet`, `efficientdet_lite0`,
   `yolov8n_float32`, and `pose_landmark_heavy`;
3. rejection of a model for which Core ML claims zero nodes, including the
   stale-counter regression case;
4. rejection of an NPU+GPU request.

The suite proves framework packaging, symbol retention, accelerator
registration, delegate ordering, Core ML conversion, inference, and fallback
diagnostics. It does not prove that any operation ran on ANE hardware; the
physical-device run under [Status](#status) is what exercised real Neural
Engine hardware.

## Reproducible build

`.github/workflows/build-coreml-ios.yml` applies both
`patches/coreml_mean_padding.patch` and
`patches/litert_coreml_npu_ios.patch` to TensorFlow v2.20.0. It builds an arm64
device framework and arm64+x86_64 simulator framework, writes the required
bundle plists, assembles the xcframework, and verifies the ordinary and
NPU-specific exported symbols.
