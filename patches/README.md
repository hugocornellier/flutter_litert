# Patches applied to TensorFlow source before building delegate binaries

Each patch here is applied by the corresponding `.github/workflows/build-*.yml`
after the TensorFlow clone and before `configure.py`. Keep them minimal and
document why, because they have to be rebased on every `TF_VERSION` bump.

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
