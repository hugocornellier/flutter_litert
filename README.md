<h1 align="center">flutter_litert</h1>

<p align="center">
<a href="https://flutter.dev"><img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform"></a>
<a href="https://dart.dev"><img src="https://img.shields.io/badge/language-Dart-blue" alt="Language: Dart"></a>
<br>
<a href="https://pub.dev/packages/flutter_litert"><img src="https://img.shields.io/pub/v/flutter_litert?label=pub.dev&labelColor=333940&logo=dart" alt="Pub Version"></a>
<a href="https://pub.dev/packages/flutter_litert/score"><img src="https://img.shields.io/pub/points/flutter_litert?color=2E8B57&label=pub%20points" alt="pub points"></a>
<a href="https://github.com/hugocornellier/flutter_litert/actions/workflows/flutter-ci.yml"><img src="https://github.com/hugocornellier/flutter_litert/actions/workflows/flutter-ci.yml/badge.svg" alt="Flutter CI"></a>
<a href="https://github.com/hugocornellier/flutter_litert/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-007A88.svg?logo=apache" alt="License"></a>
</p>

A Flutter plugin for on-device ML inference using LiteRT (formerly TensorFlow Lite), with native libraries bundled automatically on every platform.
 
## Background  

This project started as a fork of [`tflite_flutter`](https://pub.dev/packages/tflite_flutter), the official TensorFlow Lite plugin for Flutter. TensorFlow Lite has since been discontinued by Google and rebranded as [LiteRT](https://ai.google.dev/edge/litert).
 
`flutter_litert` maintains the same API as `tflite_flutter` while pre-bundling native libraries for all platforms. 

## Why this package?

The biggest pain point with `tflite_flutter` was native library setup. You had to manually build `.so`, `.dll`, or `.dylib` files and place them in the right directories for each platform. This was tedious, error-prone, and easy to get wrong.

**`flutter_litert` bundles all native libraries automatically.** Simply add the dependency, and it works out of the box.

Main improvements over `tflite_flutter`:

- Native libraries bundled automatically
  - Prebuilt binaries for macOS/Windows/Linux are served automatically. Manual steps no longer necessary.
- Native libraries are kept up to date across all platforms — [See library info](#platform-support)
- [On-device training with weight persistence](#on-device-training)
- [Custom ops support](#custom-ops)
- [Web support](#web-support)

## Installation

```yaml
dependencies:
  flutter_litert: ^1.0.3
```

That's it for native platforms. For web, call `initializeWeb()` before creating an interpreter (see [Web support](#web-support)).

## Usage

```dart
import 'package:flutter_litert/flutter_litert.dart';

final interpreter = await Interpreter.fromAsset('model.tflite');

// Prepare input and output buffers
var input = [/* your input data */];
var output = List.filled(outputSize, 0.0).reshape([1, outputSize]);

interpreter.run(input, output);
```

For inference off the main thread (native platforms):

```dart
final interpreter = await Interpreter.fromAsset('model.tflite');
final isolateInterpreter = await IsolateInterpreter.create(address: interpreter.address);

await isolateInterpreter.run(input, output);
```

## On-device training

`flutter_litert` supports [on-device training](https://ai.google.dev/edge/litert/examples/on_device_training/overview) via `SignatureRunner`, which lets you call named entry points (signatures) in a TFLite model. On-device training adjusts an existing model's weights using new data — the `.tflite` model architecture is fixed at export time and is never modified on-device.

Two persistence approaches are supported:

1. **Lightweight (`get_weights`/`set_weights`)** — Weights are extracted via builtin ops and serialized in Dart. Works with the standard bundled library on all platforms — no Flex delegate or extra downloads required.
2. **Checkpoint-based (`save`/`restore`)** — Google's standard approach using `tf.raw_ops.Save`/`Restore` with `SELECT_TF_OPS`. Writes TF V1 checkpoint files directly from the model. Requires the [Flex delegate](#flexdelegate-for-complex-model-training).

### Lightweight persistence (get_weights/set_weights)

A training-capable model using this approach exposes four signatures: `train`, `infer`, `get_weights`, and `set_weights`.

#### Preparing a training model (Python)

Export a TensorFlow model with named signatures:

```python
class MyModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable([[0.0]], dtype=tf.float32)
        self.b = tf.Variable([0.0], dtype=tf.float32)

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.float32),
        tf.TensorSpec([1, 1], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            pred = tf.matmul(x, self.w) + self.b
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(0.01 * grads[0])
        self.b.assign_sub(0.01 * grads[1])
        return {'loss': loss}

    @tf.function(input_signature=[tf.TensorSpec([1, 1], tf.float32)])
    def infer(self, x):
        return {'output': tf.matmul(x, self.w) + self.b}

    @tf.function(input_signature=[])
    def get_weights(self):
        return {'w': self.w.read_value(), 'b': self.b.read_value()}

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.float32),
        tf.TensorSpec([1], tf.float32),
    ])
    def set_weights(self, w, b):
        self.w.assign(w)
        self.b.assign(b)
        return {'w': self.w.read_value(), 'b': self.b.read_value()}
```

Convert with `TFLITE_BUILTINS` only — no Flex delegate or `SELECT_TF_OPS` needed:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
```

> **Important:** `set_weights` must return the assigned values (via `read_value()`) so the TFLite converter doesn't dead-code-eliminate the `AssignVariable` ops.

See `scripts/generate_training_model.py` for a complete working example.

#### Training loop (Dart)

```dart
final interpreter = await Interpreter.fromAsset('training_model.tflite');

// Train
final trainRunner = interpreter.getSignatureRunner('train');
final loss = Float32List(1);
for (int i = 0; i < 100; i++) {
  trainRunner.run({'x': [[inputValue]], 'y': [[targetValue]]}, {'loss': loss});
  print('Step $i, loss: ${loss[0]}');
}
trainRunner.close();

// Infer with trained weights
final inferRunner = interpreter.getSignatureRunner('infer');
final output = [[0.0]];
inferRunner.run({'x': [[inputValue]]}, {'output': output});
print('Prediction: ${output[0][0]}');
inferRunner.close();
```

#### Persisting trained weights across app sessions

The `.tflite` model file is read-only — trained weights live in memory and are lost when the interpreter is closed. Use `get_weights` and `set_weights` to persist them:

```dart
// After training — save weights to disk
final getRunner = interpreter.getSignatureRunner('get_weights');
final w = [[0.0]];
final b = [0.0];
getRunner.run({}, {'w': w, 'b': b});
getRunner.close();

final file = File('${appDocDir.path}/weights.json');
await file.writeAsString(jsonEncode({'w': w, 'b': b}));
```

```dart
// On next app launch — restore weights
final saved = jsonDecode(await File('${appDocDir.path}/weights.json').readAsString());
final setRunner = interpreter.getSignatureRunner('set_weights');
setRunner.run({'w': saved['w'], 'b': saved['b']}, {});
setRunner.close();

// Model is now in the same trained state as before
```

This uses only TFLite builtin ops (`ReadVariable`, `AssignVariable`) — no Flex delegate, no extra native libraries, works with the standard bundled library on all platforms.

### Checkpoint-based persistence (save/restore)

Google's standard approach to on-device training persistence uses `tf.raw_ops.Save` and `tf.raw_ops.Restore` with `SELECT_TF_OPS`. This writes TensorFlow V1 checkpoint files (`.index` + `.data-00000-of-00001`) directly from the model. This approach requires the Flex delegate.

#### Preparing a save/restore model (Python)

Export a model with `save` and `restore` signatures that take a checkpoint path string:

```python
class MyModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable([[0.0]], dtype=tf.float32, name='weight')
        self.b = tf.Variable([0.0], dtype=tf.float32, name='bias')

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.float32),
        tf.TensorSpec([1, 1], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            pred = tf.matmul(x, self.w) + self.b
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(0.01 * grads[0])
        self.b.assign_sub(0.01 * grads[1])
        return {'loss': loss}

    @tf.function(input_signature=[tf.TensorSpec([1, 1], tf.float32)])
    def infer(self, x):
        return {'output': tf.matmul(x, self.w) + self.b}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.string, name='checkpoint_path'),
    ])
    def save(self, checkpoint_path):
        tf.raw_ops.Save(
            filename=checkpoint_path[0],
            tensor_names=[tf.constant('weight'), tf.constant('bias')],
            data=[self.w.read_value(), self.b.read_value()],
        )
        return {'status': tf.constant(0, dtype=tf.int32)}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.string, name='checkpoint_path'),
    ])
    def restore(self, checkpoint_path):
        restored_w = tf.raw_ops.Restore(
            file_pattern=checkpoint_path[0],
            tensor_name=tf.constant('weight'),
            dt=tf.float32,
        )
        restored_b = tf.raw_ops.Restore(
            file_pattern=checkpoint_path[0],
            tensor_name=tf.constant('bias'),
            dt=tf.float32,
        )
        self.w.assign(tf.reshape(restored_w, [1, 1]))
        self.b.assign(tf.reshape(restored_b, [1]))
        return {'status': tf.constant(0, dtype=tf.int32)}
```

Convert with `SELECT_TF_OPS` enabled:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
```

> **Important:** The `save` and `restore` signatures must return a value (e.g. `status`) to prevent the TFLite converter from dead-code-eliminating the Save/Restore ops.

See `scripts/generate_training_model_flex.py` for a complete working example.

#### Save/restore in Dart

```dart
// Requires flutter_litert_flex in pubspec.yaml
final options = InterpreterOptions();
options.addDelegate(FlexDelegate());
final interpreter = Interpreter.fromFile(model, options: options);

// Train
final train = interpreter.getSignatureRunner('train');
for (int i = 0; i < 100; i++) {
  train.run({'x': [[value]], 'y': [[target]]}, {'loss': loss});
}
train.close();

// Save checkpoint to disk
final save = interpreter.getSignatureRunner('save');
save.run({'checkpoint_path': ['${appDocDir.path}/model.ckpt']}, {'status': status});
save.close();
```

```dart
// On next app launch — restore from checkpoint
final options = InterpreterOptions();
options.addDelegate(FlexDelegate());
final interpreter = Interpreter.fromFile(model, options: options);

final restore = interpreter.getSignatureRunner('restore');
restore.run({'checkpoint_path': ['${appDocDir.path}/model.ckpt']}, {'status': status});
restore.close();

// Model weights are now restored — ready for inference or continued training
```

#### Choosing a persistence approach

| | Lightweight (`get_weights`/`set_weights`) | Checkpoint (`save`/`restore`) |
|---|---|---|
| **Extra download** | None | Flex delegate (~123 MB) |
| **File format** | JSON (or any Dart serialization) | TF V1 checkpoint (`.index` + `.data`) |
| **Ops required** | `TFLITE_BUILTINS` only | `SELECT_TF_OPS` |
| **Best for** | Simple models, size-constrained apps | Google-standard models, complex architectures |
| **Model prep** | `get_weights`/`set_weights` signatures | `save`/`restore` signatures with `tf.raw_ops` |

### FlexDelegate for complex model training

The weight persistence approach above works with any model using only TFLite builtins. However, training models with layers like `Conv2D` or `BatchNormalization` generates gradient ops (e.g., `Conv2DBackpropFilter`) that require `SELECT_TF_OPS`. For these models, you need the **Flex delegate** — a separate native library (~123-492 MB per platform).

Add [`flutter_litert_flex`](https://pub.dev/packages/flutter_litert_flex) to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_litert: ^1.0.3
  flutter_litert_flex: ^0.0.1
```

That's it. The native library is downloaded automatically on the first build for all platforms. Then use the delegate:

```dart
final options = InterpreterOptions();
options.addDelegate(FlexDelegate());
final interpreter = Interpreter.fromFile(model, options: options);
```

> **Note:** Dense-only models (linear regression, MLP classifiers) do not need the Flex delegate — their gradient ops decompose into TFLite builtins. The Flex delegate is only needed when training convolutional or batch-normalized layers.

## Platform support

| Platform | Runtime | Version   | Bundling |
|----------|---------|-----------|----------|
| Android | LiteRT | 1.4.1     | Maven dependency, built automatically via Gradle |
| iOS | TensorFlow Lite | 2.20.0    | Vendored xcframeworks, linked via CocoaPods |
| macOS | TensorFlow Lite (C API) | 2.20.0    | Pre-built dylib, bundled via CocoaPods |
| Windows | TensorFlow Lite (C API) | 2.20.0    | DLL bundled via CMake |
| Linux | TensorFlow Lite (C API) | 2.20.0    | Shared library bundled via CMake |
| Web | TFLite.js (WASM via TensorFlow.js) | `tflite-js@v0.0.1-alpha.10` (default CDN) | JS runtime loaded at startup via `initializeWeb()` |

iOS and macOS will be migrated to LiteRT as official CocoaPods artifacts become available.

## Web support

`flutter_litert` supports Flutter Web, but there are a few differences from native platforms.

### Quick start (web)

Call `initializeWeb()` before creating any interpreter in a browser. It is a no-op on native, so you can call it unconditionally.

```dart
import 'package:flutter_litert/flutter_litert.dart';

await initializeWeb();

final interpreter = await Interpreter.fromAsset('assets/model.tflite');
// or: final interpreter = await Interpreter.fromBytes(modelBytes);

interpreter.run(input, output);
```

By default, `initializeWeb()` loads the TFLite.js / TensorFlow.js scripts from a CDN. You can pass custom script URLs to self-host the files (for offline use or stricter CSP).

### Web-specific API differences

- Call `initializeWeb()` before `Interpreter.fromAsset(...)` or `Interpreter.fromBytes(...)`.
- `Interpreter.fromAsset(...)` and `Interpreter.fromBytes(...)` are the supported model-loading APIs on web.
- `Interpreter.fromFile(...)`, `Interpreter.fromBuffer(...)`, and `Interpreter.fromAddress(...)` are not supported on web.
- `IsolateInterpreter.create(address: ...)` is not supported on web. Use the regular `Interpreter` directly (or `IsolateInterpreter.createFromInterpreter(...)`).
- Delegate and interpreter tuning options (GPU/XNNPACK/CoreML/threads) are accepted for API compatibility but are effectively no-ops on web.

### Using this from a web app or plugin

- Avoid `dart:io`-only code paths in the browser.
- Load files/images/models as bytes (`Uint8List`) using Flutter assets, HTTP, file picker, or drag-and-drop.
- Run your app with `flutter run -d chrome` and build with `flutter build web`.
- If you are writing a plugin on top of `flutter_litert`, add a web code path that works with bytes instead of file paths / native handles.

## Features

- **Same API as tflite_flutter.** Drop-in replacement with no code changes needed.
- **Auto-bundled native libraries.** Works out of the box on Android, iOS, macOS, Windows, and Linux (plus web support via `initializeWeb()`).
- **GPU acceleration.** Metal delegate on iOS and macOS, GPU delegate on Android, XNNPACK on all native platforms — [See delegates](#delegates).
- **CoreML delegate.** Available on iOS for Neural Engine acceleration — [See delegates](#delegates).
- **Custom ops.** MediaPipe's `Convolution2DTransposeBias` op is built and included on all platforms.
- **Isolate support.** Run inference on a background thread with `IsolateInterpreter` on native platforms (web provides a compatibility wrapper).

## Delegates

Delegates accelerate inference by offloading computation to specialized hardware (GPU, Neural Engine, etc.). All delegates are passed to the interpreter via `InterpreterOptions.addDelegate()`:

```dart
final options = InterpreterOptions();
options.addDelegate(XNNPackDelegate());
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

### Delegate availability

| Delegate | Platform | Hardware | Class |
|----------|----------|----------|-------|
| XNNPACK | Android, iOS, macOS, Windows, Linux | CPU (optimized SIMD) | `XNNPackDelegate` |
| GPU (Android) | Android | GPU (OpenGL / OpenCL) | `GpuDelegateV2` |
| Metal | iOS, macOS | GPU (Metal) | `GpuDelegate` |
| CoreML | iOS | Neural Engine / GPU / CPU | `CoreMlDelegate` |
| Flex | Android, iOS, macOS, Windows, Linux | CPU (TensorFlow ops) | `FlexDelegate` |

### XNNPACK (all native platforms)

XNNPACK is a CPU delegate that uses SIMD instructions for faster inference. It works on every native platform and is a good default accelerator.

```dart
final options = InterpreterOptions();
options.addDelegate(XNNPackDelegate(
  options: XNNPackDelegateOptions(numThreads: 4),
));
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

XNNPACK options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numThreads` | `int` | `1` | Number of threads for parallel computation |
| `flags` | `int` | `0` | Bitmask of `TfLiteXNNPackDelegateFlags` (QS8, QU8, FORCE_FP16, etc.) |
| `weightCacheFilePath` | `String?` | `null` | Path to cache packed weights on disk for faster subsequent loads |

Weight caching example:

```dart
final cacheDir = await getApplicationSupportDirectory();
final options = InterpreterOptions();
options.addDelegate(XNNPackDelegate(
  options: XNNPackDelegateOptions(
    numThreads: 4,
    weightCacheFilePath: '${cacheDir.path}/xnnpack_cache.bin',
  ),
));
```

### GPU delegate (Android)

The Android GPU delegate uses OpenGL ES or OpenCL for GPU-accelerated inference.

```dart
final options = InterpreterOptions();
options.addDelegate(GpuDelegateV2());
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

> **Note:** GPU delegate initialization on Android can take several seconds on first run as GPU kernels are compiled. Use serialization caching (below) to eliminate this overhead on subsequent runs.

#### GPU kernel serialization (Android)

Compiled GPU kernels can be cached to disk so initialization is near-instant after the first run:

```dart
final cacheDir = await getApplicationSupportDirectory();
final options = InterpreterOptions();
options.addDelegate(GpuDelegateV2(
  options: GpuDelegateOptionsV2(
    serializationDir: cacheDir.path,
    modelToken: 'my_model_v1',
    experimentalFlags: [
      TfLiteGpuExperimentalFlags.TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT,
      TfLiteGpuExperimentalFlags.TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION,
    ],
  ),
));
```

GPU delegate options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `isPrecisionLossAllowed` | `bool` | `false` | Allow FP16 quantization for performance |
| `inferencePreference` | `int` | `FAST_SINGLE_ANSWER` | `TfLiteGpuInferenceUsage` value |
| `inferencePriority1/2/3` | `int` | `MAX_PRECISION, AUTO, AUTO` | Ordered `TfLiteGpuInferencePriority` values |
| `experimentalFlags` | `List<int>` | `[ENABLE_QUANT]` | `TfLiteGpuExperimentalFlags` values |
| `maxDelegatePartitions` | `int` | `1` | Max graph partitions delegated to GPU |
| `serializationDir` | `String?` | `null` | Directory for kernel cache (requires `ENABLE_SERIALIZATION` flag) |
| `modelToken` | `String?` | `null` | Unique model identifier for cache namespace |

### Metal delegate (iOS and macOS)

The Metal delegate uses Apple's Metal API for GPU-accelerated inference. On iOS, Metal is bundled automatically. On macOS, it requires a one-time download (~20 MB).

**iOS** — works out of the box:

```dart
final options = InterpreterOptions();
options.addDelegate(GpuDelegate());
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

**macOS** — download once during development, then use identically:

```dart
// One-time download (cached locally, auto-bundled into app on next build)
await GpuDelegate.download();
```

After downloading, run `pod install` in your app's `macos/` directory to pick up the library. Then use `GpuDelegate()` the same as iOS.

> **macOS note:** The Metal delegate requires Apple Silicon (arm64). Benchmarks show **~3.4x faster** inference than XNNPACK on M-series chips (MobileNet V1: 2.7ms Metal vs 9.1ms XNNPACK 4-thread on M1).

You can also set `TFLITE_METAL_PATH` to point to a local copy of the dylib:

```bash
TFLITE_METAL_PATH=/path/to/libtensorflowlite_gpu-mac.dylib flutter run
```

Metal delegate options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allowPrecisionLoss` | `bool` | `false` | Allow FP16 for performance |
| `waitType` | `int` | `Passive` | `TFLGpuDelegateWaitType` value (Passive, Active, DoNotWait, Aggressive) |
| `enableQuantization` | `bool` | `true` | Enable quantized model support |

### CoreML delegate (iOS)

The CoreML delegate uses Apple's CoreML framework, which can dispatch to the Neural Engine, GPU, or CPU depending on the model and device.

```dart
final options = InterpreterOptions();
options.addDelegate(CoreMlDelegate(
  options: CoreMlDelegateOptions(
    enabledDevices: TfLiteCoreMlDelegateEnabledDevices
        .TfLiteCoreMlDelegateDevicesWithNeuralEngine,
  ),
));
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

CoreML delegate options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabledDevices` | `int` | `DevicesWithNeuralEngine` | Which devices to use (`AllDevices` or `DevicesWithNeuralEngine`) |
| `coremlVersion` | `int` | `0` | CoreML version to target (0 = latest available) |
| `maxDelegatedPartitions` | `int` | `0` | Max partitions (0 = unlimited) |
| `minNodesPerPartition` | `int` | `2` | Minimum nodes per delegated partition |

### Platform recommendations

| Platform | Recommended delegate | Notes |
|----------|---------------------|-------|
| Android | `XNNPackDelegate` | Safe default. `GpuDelegateV2` is faster for large models but has slow first-run init — use serialization caching to mitigate. |
| iOS | `GpuDelegate` (Metal) | Best general performance. Add `CoreMlDelegate` for Neural Engine models. |
| macOS | `GpuDelegate` (Metal) | ~3.4x faster than XNNPACK on Apple Silicon. Requires one-time `GpuDelegate.download()`. Falls back to `XNNPackDelegate` on Intel Macs. |
| Windows | `XNNPackDelegate` | XNNPACK symbols are bundled in the DLL. |
| Linux | `XNNPackDelegate` | XNNPACK symbols are bundled in the shared library. |
| Web | None needed | Delegates are no-ops on web. The WASM runtime handles optimization internally. |

## Custom ops

`flutter_litert` bundles MediaPipe's `Convolution2DTransposeBias` custom op out of the box. To use it, call `addMediaPipeCustomOps()` on your interpreter options before creating the interpreter:

```dart
final options = InterpreterOptions();
options.addMediaPipeCustomOps();
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

This is required for models like MediaPipe Selfie Segmentation (the binary `selfie_segmenter.tflite` and `selfie_segmenter_landscape.tflite` variants). The [`face_detection_tflite`](https://pub.dev/packages/face_detection_tflite) package uses this for its selfie segmentation feature.

### Adding your own custom ops

If your TFLite model uses a custom op that isn't already bundled, you need to provide three things: a C implementation, per-platform native builds, and Dart FFI registration. The bundled `Convolution2DTransposeBias` op (in `src/custom_ops/`) serves as a complete working example.

#### 1. Write the C implementation

Implement the four TFLite op callbacks and export a registration function:

```c
#include "tensorflow_lite/common.h"
#include "tensorflow_lite/c_api.h"

static void* MyOpInit(TfLiteContext* context, const char* buffer, size_t length) {
    // Parse custom_options, allocate state. Return a pointer to your state.
}

static void MyOpFree(TfLiteContext* context, void* buffer) {
    // Free state allocated in Init.
}

static TfLiteStatus MyOpPrepare(TfLiteContext* context, TfLiteNode* node) {
    // Validate input/output tensor shapes, types, and dimensions.
    // Do NOT call context->ResizeTensor for custom ops — validate
    // against the shapes the model graph already defines.
    return kTfLiteOk;
}

static TfLiteStatus MyOpEval(TfLiteContext* context, TfLiteNode* node) {
    // Run the actual computation.
    return kTfLiteOk;
}

static TfLiteRegistration g_registration = {
    MyOpInit,
    MyOpFree,
    MyOpPrepare,
    MyOpEval,
    NULL,                   // profiling_string
    kTfLiteBuiltinCustom,   // builtin_code
    "MyCustomOpName",       // custom_name (must match the op name in your .tflite model)
    1,                      // version
    NULL,                   // registration_external
};

// Export with visibility so the linker doesn't strip it and FFI can find it
__attribute__((used, visibility("default")))
TfLiteRegistration* MyPlugin_RegisterMyCustomOp(void) {
    return &g_registration;
}
```

#### 2. Build and bundle per platform

Each platform needs to compile your C code and make the resulting library available at runtime.

**Android** — Add a CMakeLists.txt that compiles your `.c` into a shared library, and point to it from your plugin's `android/build.gradle`:

```gradle
android {
    externalNativeBuild {
        cmake { path "../src/CMakeLists.txt" }
    }
}
```

**Linux / Windows** — In your plugin's `linux/CMakeLists.txt` or `windows/CMakeLists.txt`, add your source directory as a subdirectory and include the resulting library in `bundled_libraries`:

```cmake
add_subdirectory("../src" "${CMAKE_CURRENT_BINARY_DIR}/my_custom_ops")
set(my_plugin_bundled_libraries $<TARGET_FILE:my_custom_ops> PARENT_SCOPE)
```

**macOS** — Either pre-build a universal `.dylib` and ship it as a CocoaPods resource in your `.podspec`:

```ruby
s.resources = ['my_custom_ops.dylib']
```

Or compile from source using a script phase.

**iOS** — Static linking is required. Create a forwarder `.c` file in `ios/Classes/` that `#include`s your implementation:

```c
// ios/Classes/my_custom_ops.c
#include "../../src/my_custom_op.c"

// Force-load so the linker doesn't strip the symbol
__attribute__((used))
void MyPlugin_ForceLoadCustomOps(void) {
    (void)MyPlugin_RegisterMyCustomOp;
}
```

Then call the force-load function from your Swift/ObjC plugin registration to prevent dead code elimination.

#### 3. Register from Dart via FFI

Load the native library and register the op with the interpreter options:

```dart
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter_litert/flutter_litert.dart';

// Load the native library (platform-specific)
final DynamicLibrary customOpsLib = Platform.isIOS
    ? DynamicLibrary.process()  // iOS: statically linked
    : DynamicLibrary.open('libmy_custom_ops.so');  // Android/Linux/etc.

// Look up the registration function
final registerFn = customOpsLib.lookupFunction<
    Pointer<TfLiteRegistration> Function(),
    Pointer<TfLiteRegistration> Function()
>('MyPlugin_RegisterMyCustomOp');

final registration = registerFn();

// Keep this alive for the lifetime of the interpreter — TFLite stores
// the pointer, not a copy
final opName = 'MyCustomOpName'.toNativeUtf8().cast<Char>();

// Register before creating the interpreter
final options = InterpreterOptions();
tfliteBinding.TfLiteInterpreterOptionsAddCustomOp(
    options.base,  // the underlying native pointer
    opName,
    registration,
    1,  // min_version
    1,  // max_version
);
final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
```

### Gotchas

- **The op name string must outlive the interpreter.** `TfLiteInterpreterOptionsAddCustomOp` stores the pointer, not a copy. Allocate it once with `toNativeUtf8()` and keep it alive statically (e.g. as a `static Pointer<Char>?` field).
- **iOS linker stripping.** Even if the C symbol is compiled in, the linker will strip it if nothing references it. You need a force-load function called from your plugin's Swift/ObjC registration code.
- **Windows CRT heap mismatch.** If your custom op DLL calls `malloc` but TFLite frees with its own `free` (from a different DLL), you get heap corruption. Resolve `TfLiteIntArrayCreate` from the TFLite DLL at runtime so allocations use TFLite's heap. See `src/custom_ops/transpose_conv_bias.c` for a working example.
- **Web is not supported.** The TFLite.js/WASM runtime does not have a custom op registration API.

## Credits

Based on [`tflite_flutter`](https://pub.dev/packages/tflite_flutter) by the TensorFlow team and contributors.
