# iOS CompiledModel NPU

## Checkpoint status

The iOS implementation is complete through simulator validation. The simulator
suite passes, but this is not yet physical-device validation: an iOS simulator
has no Apple Neural Engine and Core ML executes the delegated model on CPU.

At this checkpoint:

- the CocoaPods-vendored `TensorFlowLiteCCoreML.xcframework` contains the
  patched device and universal simulator slices;
- SwiftPM still points at the previous released Core ML artifact, which does
  not contain the NPU entry points. Ordinary SwiftPM builds continue to work,
  but an NPU request reports unsupported until a replacement artifact is
  published and its URL and checksum are updated;
- physical-iPhone correctness and performance validation remain pending.

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
diagnostics. It does not prove that any operation ran on ANE hardware.

## Reproducible build

`.github/workflows/build-coreml-ios.yml` applies both
`patches/coreml_mean_padding.patch` and
`patches/litert_coreml_npu_ios.patch` to TensorFlow v2.20.0. It builds an arm64
device framework and arm64+x86_64 simulator framework, writes the required
bundle plists, assembles the xcframework, and verifies the ordinary and
NPU-specific exported symbols.
