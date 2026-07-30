# Checks for the NPU / Core ML accelerator work

You are working on the in-progress NPU accelerator in `flutter_litert`: `macos/coreml_npu/`,
`libLiteRtCoreMlNpuAccelerator.dylib`, `libtensorflowlite_coreml_npu-mac.dylib`, and the
`Accelerator.npu` plumbing in `lib/src/bindings/litert_loader.dart` and
`lib/src/compiled_model/compiled_model_native.dart`.

A parallel session was measuring delegate performance for the dog/cat landmark models and
turned up things that bear directly on that work. This is a list of concrete checks, each
with the evidence behind it, so you can confirm or refute them rather than take them on
faith.

---

## 1. Your Core ML delegate build almost certainly has the MEAN padding bug

**This is the most important item.** `PoolingLayerBuilder::Build()` in
`tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.cc` handles `MEAN` as
global average pooling and returns *before* the block that sets the padding type:

```cpp
if (pooling_type_ == kTfLiteBuiltinMean) {
  pooling_params->set_type(AVERAGE);
  pooling_params->set_globalpooling(true);
  return layer_.release();          // returns before padding is set
}
...
if (params->padding == kTfLitePaddingSame) {   // never reached for MEAN
  pooling_params->mutable_same();
} else {
  pooling_params->mutable_valid();
}
```

`PoolingLayerParams.PaddingType` is a required oneof, so Core ML's validator rejects the
**entire model** at compile time:

```
Error compiling model compiler error: Error reading protobuf spec. validator error:
  Padding type for the pooling layer 'PoolingLayerBuilder (MEAN)_32' is not set.
ERROR: Failed to Compile and save Model.
ERROR: CoreMl Kernel was not initialized
ERROR: Node number 279 (TfLiteCoreMlDelegate) failed to prepare.
ERROR: Restored original execution plan after delegate application failure.
```

TFLite then restores the CPU plan, so **nothing crashes and no error reaches the caller.**
The delegate simply never runs. That is why `doc/delegate_verification.md` recorded
`dev = 0.0` for CoreML across every model it tested: one unset protobuf field, not five
separate failures.

The fix is `patches/coreml_mean_padding.patch` in this repo (one line,
`pooling_params->mutable_valid();` before the early return), already wired into
`build-coreml-macos.yml` and `build-coreml-ios.yml`.

**Checks for you:**

- Determine what source produced `libtensorflowlite_coreml_npu-mac.dylib`. If it is TFLite's
  Core ML delegate at any revision that still has this early return, it has the bug.
  Upstream has not fixed it as of `v2.20.0`.
- If your NPU path routes through a Core ML delegate at all, apply the patch to that build
  too. A CPU+NeuralEngine delegate that cannot compile a `MEAN` op will silently run on CPU.
- **7 of the 8 models shipped across `dog_detection`, `cat_detection` and `animal_detection`
  contain `MEAN`.** Only `superanimal_hrnet_w32` does not. Measured op counts:

  | model | MEAN ops |
  |---|---|
  | species_classifier | 10 |
  | rtmpose_s (default pose) | 4 |
  | ssdlite | 8 |
  | dog_face_localizer | 24 |
  | cat_face_localizer | 24 |
  | dog_face_landmarks | 8 |
  | cat_face_landmarks | 8 |
  | hrnet_w32 | **0** |

  So `hrnet_w32` is the only model that would appear to work if the bug is present, which
  makes it a misleading smoke test. **Test with a MEAN-containing model** or you will
  conclude the delegate works when it does not.

---

## 2. Verify engagement by output deviation, never by latency

A delegate that attaches, claims zero ops and falls back to CPU looks exactly like a slow
success. Timing cannot distinguish "ran on the ANE and was slow" from "never ran on the ANE".

The only reliable test, which `doc/delegate_verification.md` already uses:

1. Run the model with no delegate, keep the output as a reference.
2. Run it with your accelerator on the **same input**.
3. Compare: `dev == 0.0` exactly means the accelerator did nothing. Small `dev` (~1e-3 for
   fp16 ANE) means engaged. Large `dev` (~1e-1) means engaged and computing wrong answers.

Two traps that produced false results in the parallel session:

- **Compute `dev` from a single invocation on a fixed input.** Comparing the last output of
  two timing loops with different iteration counts compares different images. That reported
  `dev = 7.5e-01` for a delegate whose true `dev` was `5.4e-07`, which looks exactly like the
  upstream corruption signature and would have been filed as one.
- **Do not import TensorFlow into a process that drives these dylibs.** TF ships its own copy
  of the TFLite and Core ML delegate symbols; with both loaded, Core ML delegate creation goes
  from 0.1 s to hanging past 120 s. Measured A/B with an `import tensorflow` as the only
  difference. This produced a bogus "CoreML hangs for 10+ minutes" finding that was retracted.

---

## 3. `CoreMlDelegateOptions` defaults are not all zero

If you hand-build the options struct rather than using the Dart factory:

- Field order is four consecutive ints: `enabled_devices`, `coreml_version`,
  `max_delegated_partitions`, `min_nodes_per_partition`.
- `CoreMlDelegateOptions` defaults `minNodesPerPartition` to **2**, and
  `_createCoreml` overrides only `enabledDevices`. Zeroing `min_nodes_per_partition` lets Core
  ML create a partition per node, which on a 279-op model is a very large number of tiny
  compilations.
- `coremlVersion` defaults to **0**, which is not a valid value, so every delegate creation
  logs `coreml_version must be 2 or 3. Setting to 3.` Harmless but noisy on each init;
  defaulting to 3 explicitly would silence it.

---

## 4. Do not extrapolate device performance from this Mac. It failed badly.

The parallel session patched the GPU delegate to accept `TRANSPOSE_CONV` v4 (see
`patches/gpu_transpose_conv_v4.patch` and `doc/graph_shape_vs_delegate.md`) and measured, over
all 480 DogFLW test images:

| landmark model | M4 Max (40 GPU cores) | iPhone 15 Pro (6 GPU cores) |
|---|---|---|
| Metal GPU | **5.11 ms** | **46.90 ms** |
| XNNPACK, 4 threads | 26.83 ms | 47.41 ms |

A 5.2x win on the Mac became **1%** on the phone, because the GPU times scale with core count
while the CPU path does not. Any ANE number from macOS predicts nothing about an A17 Pro.
Measure on device.

Relatedly, `PerformanceConfig`'s doc comment advertising "XNNPACK (2-5x SIMD acceleration)"
measured **1.18x** on the landmark model.

---

## 5. Two live, unrelated silent-fallback bugs on iOS worth knowing about

Both measured on an iPhone 15 Pro with the models as shipped:

| landmark model, iOS | median | dev vs CPU | note |
|---|---|---|---|
| XNNPACK | **47.41 ms** | 7.0e-06 | engaged |
| GPU (what auto-mode picks on iOS) | 56.82 ms | **0.0** | no-op, bare CPU |
| CoreML | 57.38 ms | **0.0** | no-op, bare CPU |

`InterpreterFactory._createAutoMode` sends iOS to the GPU delegate. On this model that is a
silent no-op, so iOS runs on bare CPU **without** XNNPACK and pays roughly 20% for nothing.
This matches the "17-23% latency, silently" that `doc/delegate_verification.md` records for
zero-op delegation.

Two things follow that may affect your design:

- Whatever `Accelerator.npu` does, it needs the same `dev != 0.0` guard, or it will join the
  list of modes that are quietly worse than the default.
- `PerformanceConfig` is currently pipeline-global in the sibling packages. Per-stage overrides
  were just added to `animal_detection` (`posePerformanceConfig`) and to `dog_detection` /
  `cat_detection` (`landmarkPerformanceConfig`) precisely because the right accelerator differs
  per model. If NPU is only good for some ops, expect callers to need that granularity.

---

## 6. Building an iOS Core ML framework from source: two gotchas

If you need an iOS build of a patched delegate, `build-coreml-ios.yml` in this repo now does
it. Two things cost time to discover:

- **`TF_CONFIGURE_IOS=1` is required.** `configure_ios()` symlinks `BUILD.apple` to `BUILD`
  for every path in `APPLE_BAZEL_FILES`, which includes `tensorflow/lite/ios/BUILD`. `v2.20.0`
  ships only `BUILD.apple` there, so without the env var bazel reports
  `no such package 'tensorflow/lite/ios': BUILD file not found`.
- **bazel's framework output has no `Info.plist`.** Xcode rejects it with
  `Framework ... did not contain an Info.plist`. One has to be written into each slice, with
  `CFBundleSupportedPlatforms` set to `iPhoneOS` for device and `iPhoneSimulator` for the
  simulator slice. The release xcframeworks have one; a fresh bazel build does not.
- SPM also rejects absolute `binaryTarget(path:)` values: *"path expected to be relative to
  package root"*. `scripts/use_local_coreml_xcframework.sh` handles staging a locally built
  xcframework inside the package for testing, and reverts cleanly.

---

## What would be genuinely useful to know

1. Does `libtensorflowlite_coreml_npu-mac.dylib` compile a `MEAN`-containing model? If not,
   apply `patches/coreml_mean_padding.patch` to its source and rebuild.
2. With that fixed, does the ANE beat **47.41 ms** on an iPhone 15 Pro for the dog landmark
   model? That is the number to beat, and it is the bar the GPU path failed.
3. Does `Accelerator.npu` detect and report zero-op delegation, or does it fall back silently
   like `gpu` and `coreml` currently do?
