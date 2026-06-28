## 3.2.0

Restores the WASM-ready score on pub.dev (back to 160/160), which dropped to
150 when pub.dev upgraded its analyzer (pana 0.23.13). pana 0.23.13 mis-resolves
conditional `export`/`import` directives: it derives the condition name with
`name.tokens.map((t) => t.value()).join()`, and because `library` is a Dart
keyword `Token.value()` returns it upper-cased, so `if (dart.library.X)` becomes
`dart.LIBRARY.X` and never matches. Every conditional then resolves to its
default (first) URI. The main `flutter_litert.dart` barrel defaulted to the
native (`dart:ffi` / `dart:isolate`) surface, so the WASM/platform analysis saw
those libraries as reachable.

* The portable `flutter_litert.dart` barrel now defaults to the WASM-safe web
  surface and gates the native surface on `dart.library.io`, so the package is
  WASM-compatible again. Runtime behavior is unchanged: real native and web
  builds resolve exactly as before.
* **Breaking (native-only):** API whose public signatures use native-only types
  (`Isolate`, `SendPort`, `File`, ...) and therefore cannot be WASM-safe is now
  published from a new `package:flutter_litert/native.dart` library instead of
  the main barrel: `IsolateWorkerBase`, `IsolateRpcClient`,
  `setupIsolateHandshake`, `InterpreterPool`, `ModelCheckpoint`. Native code
  using these now also needs `import 'package:flutter_litert/native.dart';`.
  `TensorFloat32Views` and the rest of the API stay on the main barrel.
* `InterpreterOptions` on web gains `hasDelegate`, `threads`, and
  `copyWithoutDelegates()` to match the native API.

## 3.1.4

* Preserve thread tuning and custom-op registrations when delegate application
  fails and interpreter creation retries on CPU.
* Expose whether an interpreter actually has an active delegate, so isolate
  selection follows the effective backend after fallback.

## 3.1.3

Interpreter creation now falls back to CPU when a configured delegate cannot
be applied to a model/runtime, instead of failing. This fixes classic
`Interpreter` creation for models that cannot use the default iOS Metal
delegate, including on the iOS simulator: it now warns and retries on CPU. The
fallback covers every creation path (`fromAsset`, `fromBuffer`, `fromBytes`, and
the isolate interpreter), and the iOS integration job now also exercises the
classic `Interpreter` path so this is caught in CI.

## 3.1.2

Makes the package web- and WASM-compatible. `dart:isolate` was reachable from
the public API (via `decode_failure.dart` and `isolate_rpc_server.dart`) but is
unavailable on web/WASM; the isolate-dependent code now sits behind conditional
imports so none of it is reachable on the web build. No API changes.

## 3.1.1

Fixes the iOS CocoaPods build for the LiteRT Next runtime. The prebuilt
`LiteRt.xcframework` / `LiteRtMetalAccelerator.xcframework` download shipped an
arm64-only `ios-arm64-simulator` slice, whose identifier does not match the
`ios-arm64_x86_64-simulator` slice CocoaPods selects on the simulator. The build
then failed copying a non-existent slice (`rsync ... No such file or directory`).

* Fix: the downloaded iOS frameworks now carry a universal
  `ios-arm64_x86_64-simulator` slice (arm64 device binary + x86_64 stub), so the
  simulator build resolves and links. (SwiftPM builds were unaffected.)
* Fix: the podspec now verifies the simulator slice — not just the device
  slice — before skipping the download, and clears stale slices on re-download,
  so an existing arm64-only cache is replaced.

## 3.1.0

Additive release: shared isolate, CompiledModel-pooling, and image-RPC
utilities, extracted so the packages built on flutter_litert can maintain them
in one place instead of each carrying its own copy. No breaking changes; the
`Interpreter` and `CompiledModel` APIs are unchanged.

* New: `serveIsolateRpc` — the isolate-side counterpart to `IsolateRpcClient`.
  Drives the `{id, op}` -> `{id, result | error}` protocol from a handler map,
  replacing the hand-written `listen`/`switch`/try-catch envelope each worker
  isolate used to carry. `IsolateRpcExactError` lets a handler send a verbatim
  wire-error string when the main side relies on the exact text (e.g. a
  `startsWith` error contract).
* New: `IsolateWorkerBase.disposeGracefully` and
  `IsolateRpcClient.disposeGracefully` — send the dispose op and await the
  isolate's acknowledgement before killing it, so the isolate can free native
  interpreters / `CompiledModel`s. `Isolate.kill(priority: immediate)` otherwise
  races past the queued dispose message and leaks the native handles.
* New: `CompiledModelPool` — a round-robin pool of `CompiledModel` slots, each
  with its own reusable input buffer and `AsyncLock`, so concurrent inferences
  (e.g. one per detected object) land on distinct models with leak-free init
  teardown. A pool of size 1 degrades to a safe single-model-plus-lock.
* New: `compiled_io_utils` — `compiledFloatCount`, `squareSideFromFloats`,
  `compiledSquareInputSide`, `compiledOutputFloatCounts`, and
  `indexWhereFloatCount` for deriving tensor geometry from a `CompiledModel`,
  whose tensor sizes are exposed only in bytes.
* New: `cameraFrameRpcFields` and `cameraFrameFromRpcMessage` — pack a
  `CameraFrame` into an isolate-request field map and rebuild it on the isolate
  side (any image decode stays in the consumer, keeping this dependency-free).
* New: `decodeFailurePrefix`, `throwDecodeFailure`, and
  `rethrowOrFormatException` — signal an undecodable-image failure from inside
  an isolate and surface it as a `FormatException` on the main side instead of a
  cryptic downstream error.

## 3.0.0

* New: LiteRT Next `CompiledModel` API: `CompiledModel.fromFile`,
  `CompiledModel.fromBuffer`, and `CompiledModel.fromBufferWithGpuFallback`,
  with automatic hardware-accelerator selection via
  `Accelerator.{cpu, gpu, npu}`, `Precision`, and `TensorBufferMode`. This is
  the recommended path for GPU/NPU acceleration going forward, following
  Google's LiteRT Next guidance
  (https://developers.google.com/edge/litert/next/get_started). Supported on
  Android, iOS, macOS, Windows, and Linux.
* The desktop (Windows/Linux) WebGPU GPU accelerator and DirectX Shader Compiler
  are fetched from a GitHub release at build time instead of being bundled in the
  published package, keeping it under pub.dev's 100 MiB size limit. Desktop GPU
  acceleration still works; the libraries download automatically (verified by
  SHA-256) on the first build. No effect on Android, iOS, or macOS.
* Deprecated: manual hardware-acceleration delegates for the Interpreter API,
  namely `GpuDelegateV2` (Android GL/CL), the Metal `GpuDelegate`, and
  `CoreMlDelegate`
  (with their `*Options`). They remain fully functional but are superseded by
  `CompiledModel`'s automatic accelerator selection and are planned for removal
  in 4.0.0. The Interpreter API itself, the CPU `XNNPackDelegate`, and
  `FlexDelegate` are NOT deprecated and remain fully supported.
* Fix: creating an `XNNPackDelegate` with `XNNPackDelegateOptions` no longer
  crashes on the arm64 Android emulator. The options struct was initialized by
  calling the native `TfLiteXNNPackDelegateOptionsDefault()`, which returns the
  struct by value; that by-value FFI return crashes the Dart VM on the arm64
  Android emulator (it works on real devices, macOS, and iOS). The struct is now
  built in Dart, matching upstream `tflite_flutter`, while preserving the QS8/QU8
  quantization defaults; the resulting native options are unchanged, so there is
  no behavior difference on real devices.
* Fix: `TensorFloat32Views` input views are now genuinely writable. They were
  previously built from the unmodifiable `Tensor.data` view, so indexed writes
  (`views.inputs[0][i] = x`) threw `UnsupportedError`, and bulk
  `setAll`/`setRange` only worked through a Dart VM enforcement gap that a
  future SDK could close. Views are now captured via the new
  `Tensor.asFloat32View()`, a mutable `Float32List` aliasing the tensor's
  native buffer (valid until the next resize/`allocateTensors`).
* `SignatureRunner.run()` per-call overhead roughly halved (16–17µs → 7µs per
  call on the bundled `test/benchmark/signature_runner_benchmark_test.dart`):
  tensor handles are cached by name between allocations, and the valid-names
  error text is built only when a lookup actually fails instead of on every
  `getInputTensor`/`getOutputTensor` call.
* Behavior change: `IsolateInterpreter.run`/`runForMultipleInputs` no longer
  silently drop calls. A call issued while a previous run is in flight is now
  queued and completes with real results (previously it returned normally
  without writing the output buffers); frame-skipping callers can check
  `state == IsolateInterpreterState.loading` before calling. Running after
  `close()` now throws `StateError` instead of returning silently.
* `TensorType.fromValue` is O(1) instead of scanning all enum values (it runs
  on every `Tensor.type` access), and inference timing uses a reused monotonic
  `Stopwatch` instead of two `DateTime.now()` calls per run.
* Interpreter hot-path overhaul, measured on the bundled
  `test/benchmark/engine_overhead_benchmark_test.dart` (MediaPipe
  face_detection_short_range, macOS host):
  * `run()`/`runForMultipleInputs()` with nested-list input and output drops
    from 8.9ms to 1.9ms per inference (native floor 1.0ms) by converting
    tensors through a single pre-sized buffer instead of one small allocation
    per element, and by reading outputs through typed views instead of a
    per-element `ByteData.view`.
  * `Tensor.setTo`/`copyTo` now copy directly between Dart memory and
    `TfLiteTensorData` instead of round-tripping through a native scratch
    buffer (two extra copies per tensor per inference).
  * Fix: passing a flat `Float32List` (or other typed data) as an input no
    longer resizes the input tensor to rank 1, which broke models with
    rank-sensitive ops (`CONV_2D failed to prepare`). Flat typed data whose
    element count matches the tensor is now staged as-is, and is the fastest
    `run()` input type.
  * New: outputs can be flat typed data (`Float32List`, `Int32List`,
    `Int64List`, `Int16List`, `Int8List`). Bytes are bulk-copied directly
    into the buffer; previously this threw a shape-mismatch `ArgumentError`.
    `run()` with `Float32List` in/out now measures within ~7% of the
    raw tensor-views floor.
  * Behavior note: `copyTo(Uint8List)`/`copyTo(ByteBuffer)` now fill and
    return the destination instead of returning a separate copy.
* CompiledModel: per-dispatch native out-params are allocated once per model
  instead of per call (`run`, `runAsync`, lock/unlock paths).

## 2.8.3

* Android: support both AGP 8 and AGP 9 by moving the plugin Gradle files to
  Kotlin DSL and updating the Android tooling plugin declarations (6c332e3b).

## 2.8.2

* Fix GPU and CoreML delegates silently falling back to CPU on macOS and iOS
  (#11). macOS now bundles the GPU/CoreML dylibs that were previously omitted
  from the Swift Package manifest. iOS retains all 212 packaged LiteRT, Metal,
  and CoreML symbols that Dart FFI resolves but linker stripping would otherwise
  drop. They are kept through a generated anchor on both CocoaPods and SPM, plus
  an embedded dynamic SPM framework so they survive App Store archive stripping.
  This extends the 2.8.0 mitigation (#8, #9) to every packaged API.
* Log delegate initialization failures before falling back to CPU. A packaging
  regression can no longer masquerade as unexpectedly slow GPU/CoreML
  performance.
* On Apple Silicon the Metal GPU is substantially faster for conv-heavy models
  once actually loaded (for example, selfie segmentation ~27ms -> ~3ms versus
  XNNPACK).

## 2.8.1

* Complete the AGP 9 / built-in Kotlin fix from 2.8.0 (#10). 2.8.0 resolved the
  "Inconsistent JVM-target ... (17) and (21)" error on the AGP 8.11 transitional
  path, but a full migration to `android.builtInKotlin=true` on AGP 9 still
  failed with "The 'org.jetbrains.kotlin.android' plugin is no longer required
  since AGP 9.0": the Flutter Gradle plugin auto-applies the legacy Kotlin
  plugin to this module, and AGP 9 rejects it. The plugin now applies
  `kotlin-android` only on AGP < 9 (which also stops Flutter from auto-applying
  it), and keeps the JVM-target pin guarded so the AGP-9-without-built-in-Kotlin
  case is skipped. Verified building against AGP 8.11.1 and 9.0.1 with built-in
  Kotlin both enabled and disabled.

## 2.8.0

* Fix the "Inconsistent JVM-target compatibility detected ... (17) and (21)"
  Android build failure under AGP 9 / Flutter 3.44+ (#10). The fix pins the
  Kotlin JVM target to 17 only when AGP >= 9 is in use, so older Flutter/AGP
  toolchains are unaffected and the minimum supported versions are unchanged.
* Fix iOS "Failed to lookup symbol 'TfLiteInterpreterOptionsCreate'" crash on
  App Store / TestFlight builds (#8, #9). The TFLite C symbols are resolved at
  runtime via dlsym and were stripped during App Store distribution; the
  CocoaPods podspec now disables that stripping on the app target so no manual
  Xcode build-setting changes are required.

## 2.7.0

* Add `InterpreterOptions.addCustomOp(...)`: high-level method for registering
  custom TFLite ops; handles native string allocation and lifetime internally,
  replacing the previous raw `tfliteBinding` call pattern.
* Add `Interpreter.fromBytes(Uint8List)`: async cross-platform constructor,
  matching the web API. Native platforms complete immediately; unsupported stub
  throws `UnsupportedError`.
* Rename `lastNativeInferenceDurationMicroSeconds` →
  `lastInferenceDurationMicroseconds` on `Interpreter`, `SignatureRunner`, and
  `LiteRtInterpreter` (all platform variants). Old name kept as a `@Deprecated`
  alias.
* Rename `configureLiteRtLoader` → `configureLiteRtWebLoader`. Old name kept as
  a `@Deprecated` alias and re-exported from `all_web.dart`.
* Fix `camera_frame.dart`: widen `.planes` cast from `List<dynamic>` to
  `Iterable<dynamic>` for broader compatibility.

## 2.6.0

* Fix iOS Swift Package Manager builds: repackage the bundled TensorFlowLite xcframeworks (correct simulator slice identifiers and framework structure) so they resolve under SPM, including on the iOS simulator.
* Add a standalone example that depends only on `flutter_litert` and `opencv_dart`.

## 2.5.8

* Raise minimum deployment targets to iOS 13.0 / macOS 10.15 to satisfy Swift Package Manager's `FlutterFramework` requirement (fixes SPM build failures on macOS/iOS).
* Update example and documentation to use `flutter_litert_flex: ^1.0.0`.

## 2.5.7

* Update example to use `flutter_litert_flex: ^0.0.8`.

## 2.5.6

* Fix SPM: add missing `FlutterFramework` dependency to iOS and macOS `Package.swift`.

## 2.5.5

* Add SPM support for iOS: TensorFlowLiteC, TensorFlowLiteCMetal and TensorFlowLiteCCoreML are now declared as binary targets in Package.swift so the plugin works with Flutter Swift Package Manager integration.
* Fix duplicate XNNPack symbol linker errors when flutter litert flex is used alongside flutter litert by removing XNNPack definitions from TFLiteFlex and hiding overlapping symbols in TensorFlowLiteC via nmedit.
* Fix stale flex dedup marker in podspec that caused nmedit to be skipped on re-downloaded xcframeworks.

## 2.5.4

* Fix WASM compatibility: replace dart:io import in camera_frame.dart with flutter/foundation.dart to allow package to compile under the WASM runtime.

## 2.5.3

* prepareCameraFrameFromImage and prepareCameraFrame now auto-detect isBgra based on platform. macOS uses BGRA, Windows and Linux use RGBA. The isBgra parameter is now nullable and no longer needs to be passed manually.

## 2.5.2

* Update documentation

## 2.5.1

* Add `decodeBitmap(Uint8List bytes)` free function: decodes encoded image bytes (JPEG, PNG, etc.) to a `web.ImageBitmap` via `createImageBitmap`, off the main thread.
* Add `WebGpuFallback` mixin: transparent WebGPU-to-WASM runtime fallback for web detector classes. Provides `withFallback<T>()` which catches GPU errors, swaps all runners to WASM via `swapToWasm()`, and retries once. Apply with `with WebGpuFallback`; implement `activeAccelerator` and `swapToWasm()`.
* Both exported from `package:flutter_litert/flutter_litert.dart` on web.

## 2.5.0

* Add `LiteRtInterpreter`, an alternative web inference path backed by Google's official LiteRT.js runtime (`@litertjs/core`). Selectable at construction time via `LiteRtInterpreter.fromBytes(bytes, accelerator: 'webgpu' | 'wasm')`, with automatic fallback from `webgpu` to `wasm` when ops aren't supported by the GPU delegate.
  * Surface chosen to match the `Interpreter` hot path used by detector packages: `fromBytes`, `getInputTensor` / `getOutputTensors`, `runForMultipleInputs(inputs, outputs)`. `runForMultipleInputs` is async (LiteRT.js `run` returns a `Promise`).
  * Output buffers can be supplied as `Float32List`, `ByteBuffer`, or the legacy nested `List<List<List<double>>>` shape used by tflite-js callers; the float-typed buffer paths take a single bulk copy.
  * Read paths use `JSFloat32Array.toDart` directly, skipping the `dataSync().dartify()` round-trip.
  * Faster output readback in the existing tflite-js `Interpreter._tensorFromJSTensor`: replaces `dataSync().dartify() as List<double>` + `Float32List.fromList(...)` with a single bulk copy via `JSTensorExtensions.dataSyncFloat32`. ~25 ms / call savings on a 705k-element YOLOv8n output.
  * Auto-loader: by default the first `LiteRtInterpreter.fromBytes(...)` call programmatically appends a `<script type="module">` to `<head>` that imports `@litertjs/core` from jsDelivr and calls `loadLiteRt(...)`; consumers don't have to touch their `web/index.html`. Override URLs (for self-hosting / strict CSP) or disable auto-loading via `configureLiteRtLoader(moduleUrl: ..., wasmUrl: ..., autoLoad: ...)`. Existing host-page loaders that assign `window.LiteRt` and dispatch a `litert-ready` event still work.
  * Pure additive: native and unsupported targets are unchanged; the existing tflite-js `Interpreter` remains the default web runtime.

## 2.4.1

* Make `camera_overlay.dart` WASM-compatible on Flutter Web

## 2.4.0

* Add painter primitives `drawLandmarkMarker`, `drawSkeletonConnections`, and `drawBoundingBoxOutline` for reuse by detector example apps and overlay widgets. Pure Dart + `dart:ui`, no new dependencies.

## 2.3.0

* Add camera-overlay helpers used across detector example apps: `rotationForFrame`, `detectionSize`, `coverFitScaleOffset`, `barQuarterTurns`, and `FpsCounter`. All pure Dart + Flutter SDK, no new dependencies. Lets example apps drop ~200 lines of duplicated orientation / sizing / FPS boilerplate.

## 2.2.2

* Add `prepareCameraFrameFromImage`, a duck-typed wrapper around `prepareCameraFrame` that accepts a `CameraImage`-shaped object directly (any object exposing `width`, `height`, `planes` with `bytes`/`bytesPerRow`/`bytesPerPixel`). Lets detector packages expose one-line camera-stream APIs without adding `package:camera` as a dependency here. Pure Dart, no new dependencies.

## 2.2.1

* Add `prepareCameraFrame` helper plus `CameraFrame`, `CameraFrameConversion`, and `CameraFrameRotation` types. Describes a camera frame (YUV420 or packed BGRA/RGBA) in a pure-Dart descriptor that detector packages can hand to their existing detection isolate, moving the `cvtColor` / `rotate` work off the UI thread without adding `opencv_dart` as a dependency here.
* Add `CameraPlane` typedef (structurally identical to `YuvPlane`; use whichever name reads better at the call site).
* Add `TensorFloat32Views` (native only): captures `Float32List` views of an `Interpreter`'s input/output tensors once after `allocateTensors`, letting detector packages reuse the same view wrappers on every inference instead of recreating them per-call. Pure Dart, no new dependencies.

## 2.2.0

* Add `packYuv420` helper for packing NV12 / NV21 / I420 camera frames into a contiguous buffer

## 2.1.0

* Minor performance/accuracy optimizations: 
  * Remove unnecessary rounding in `fillNHWC4D`  
  * Add direct `Float32List` fast paths for common tensor flattening shapes

## 2.0.13

* Fix Android JVM target mismatch: bump Java compile target to 17 to match Kotlin target set by Flutter toolchain

## 2.0.12

* Fix Android Flutter beta builds by aligning Kotlin and Java JVM targets to 11

## 2.0.11

* Fix edge case in output buffer allocation

## 2.0.10

* Update documentation

## 2.0.9

* Enable XNNPACK delegate on Android (ARM NEON SIMD acceleration in auto mode)
* Allow explicit `PerformanceConfig.xnnpack()` on iOS
* Initialize XNNPackDelegateOptions from native defaults (preserves QS8/QU8 quantization flags)

## 2.0.8

* Add Windows XNNPack delegate support (2-5x CPU inference speedup via SIMD)
* Add CI workflow to build Windows TFLite C DLL from source with XNNPack symbols

## 2.0.7

* Fix Android custom ops library alignment for 16 KB page-size devices

## 2.0.6

* Add useIsolateInterpreter parameter to skip nested isolate creation

## 2.0.5

* Fix native crash during repeated inference by removing unsafe output tensor writeback

## 2.0.4

* Fix macOS native crashes by disabling auto IsolateInterpreter for no-delegate interpreters.

## 2.0.3

* Fix WASM compatibility: move `dart:isolate` imports behind conditional exports so web compilation path is WASM-safe

## 2.0.2

* Fix: use-after-free when interpreter reads model weights from freed buffer, transfer buffer ownership from `Model` to `Interpreter`

## 2.0.1

* Add `IsolateWorkerBase` for shared isolate lifecycle management
* Add `RoundRobinPool` generic round-robin pool utility
* Add `TensorType` enum, `LandmarkMixin`, `listUtils` shared helpers
* Add weighted NMS with spatial grid optimization to `nms()`
* Consolidate platform-specific byte conversion into shared implementation
* Consolidate platform-specific tensor logic (native/web/unsupported)
* Consolidate desktop library loading into `DelegateLibraryLoader`
* Remove dead files: `all_unsupported.dart`, `version.dart`, `flutter_litert_method_channel.dart`, `flutter_litert_platform_interface.dart`
* Fix: `Model` buffer leak, delegate options leak, stale tensor cache

## 2.0.0

**Breaking:** `Point.x` and `Point.y` changed from `int` to `double`.

* Upgrade `Point` to double-precision with optional `z` depth, `==`/`hashCode`, `toMap()`/`fromMap()`, `is3D`
* Add shared `BoundingBox` class (4-corner Point-based, supports rotated boxes)
  * `BoundingBox.ltrb()` factory for axis-aligned boxes
  * `left`/`top`/`right`/`bottom` convenience getters
  * `width`, `height`, `center`, `corners` computed properties
  * `toMap()`/`fromMap()` serialization

## 1.4.0
* Fix tensor cache bug, add shared Point class, dedup internals

## 1.3.1
* Add NaN handling to `clamp01()`, returns 0.0 for NaN inputs

## 1.3.0
* Add `IsolateRpcClient` and `setupIsolateHandshake` for reusable isolate request/response communication

## 1.2.0
* Add shared ML utility functions
  * `sigmoid`, `sigmoidClipped`, `clip`, `clamp01`, `argSortDesc`, `median`, `normalizeRadians` (math utilities)
  * `iouXYXY`, `nms` (non-maximum suppression)
  * `computeLetterboxParams`, `computeAspectPadParams`, `LetterboxParams`, `AspectPadParams` (image preprocessing)
  * `bgrBytesToRgbFloat32`, `bgrBytesToSignedFloat32`, `fillNHWC4DFromBgrBytes` (image-to-tensor conversion)
  * `allocTensorShape`, `createOutputBuffers`, `zeroOutputBuffers`, `createNHWCTensor4D`, `fillNHWC4D`, `flattenDynamicTensor` (tensor allocation)
  * `decodeDetectionOutputs`, `transpose2D`, `concat0`, `ensure2D`, `xywhToXyxy` (model output decoding)
  * `postProcessDetections`, `Detection`, `decodeAndSplitOutputs` (end-to-end detection post-processing with NMS)

## 1.1.1
* Fix package layout to follow Pub conventions

## 1.1.0
* Add `PerformanceConfig` and `PerformanceMode`, 
* Add `InterpreterFactory` and `InterpreterPool`
* Add `generateAnchors()` and `SSDAnchorOptions`
* Add `scaleFromLetterbox()` utility for letterbox-to-original coordinate mapping

## 1.0.3
* Add `SignatureRunner` for on-device training workflows (`train`, `infer`, `get_weights`, `set_weights` signatures)
* Add Linux FlexDelegate support via `flutter_litert_flex` (Linux x86_64, built from TF 2.20.0 source). All three desktop platforms (macOS, Windows, Linux) now fully support on-device training with `SELECT_TF_OPS` models and checkpoint save/restore.
* Add `Interpreter.signatureCount`, `signatureKeys`, `getSignatureKey()`, `getSignatureRunner()`
* Add `SignatureRunner.cancel()`, `getInputTensors()`, `getOutputTensors()`, `lastNativeInferenceDurationMicroSeconds`

## 1.0.2
* Add native dylibs to SPM Package.swift 
* Update Dart loading paths for SPM bundle

## 1.0.1
* Improve Custom Ops documentation

## 1.0.0
* Upgrade Linux TFLite native library from 2.9.3 to 2.20.0 (built from source via CMake + Ninja + GCC x86_64)
* First stable release: 
  * All platforms are on updated 2.20.0 library files, official final stable release of TFLite
  * Pre-bundling works on supported native platforms: users no longer need to bundle native libraries manually as was required with `tflite_flutter`
  * Custom ops supported, see [face_detection_tflite v5.0.2](https://pub.dev/packages/face_detection_tflite/versions/5.0.2) `example` directory for a working example (the binary segmentation model selfie_segmenter.tflite uses custom ops)
  * Web support (experimental) functional, see [pose_detection v1.0.1](https://pub.dev/packages/pose_detection/versions/1.0.1) `web_example` directory for a working example

## 0.2.2
* Update dependencies

## 0.2.1
* Update documentation

## 0.2.0
* Web support (experimental)

## 0.1.16
* Register iOS pluginClass

## 0.1.15
* Add missing null check in interpreter teardown path on macOS

## 0.1.14
* Improve IsolateInterpreter shutdown reliability on iOS to prevent rare use-after-free when closing during active inference

## 0.1.13
* Add Swift Package Manager (SPM) support for iOS and macOS

## 0.1.12
* Upgrade Windows TFLite native library from 2.18.0 to 2.20.0 (built from source via CMake + Ninja + MSVC x64)

## 0.1.11
* Fix iOS: download xcframeworks at pod install time so static linking works on first build

## 0.1.10
* Fix macOS: bundle native libraries in pub package so `flutter test` works without manual setup

## 0.1.9
* Fix iOS and macOS podspec compatibility with Ruby 3.4+ (Prism parser)

## 0.1.8
* Upgrade iOS TensorFlow Lite from 2.17.0 (CocoaPods) to 2.20.0 (built from source via Bazel)
* Replace CocoaPods TensorFlowLiteSwift dependency with vendored xcframeworks (TensorFlowLiteC, Metal delegate, CoreML delegate)
* All xcframeworks support device arm64 + simulator arm64/x86_64 (Apple Silicon and Intel Macs)

## 0.1.7
* Improved documentation

## 0.1.6
* Upgrade macOS TFLite native library from 2.17.1 to 2.20.0 (latest stable, universal binary: arm64 + x86_64)
* Update all C API headers to TFLite 2.20.0
* Regenerate FFI bindings (`TfLiteOperatorCreate` now takes 4 params, `TfLiteOperatorCreateWithData` removed, new `kTfLiteOutputShapeNotKnown` status, new builtin ops)
* Rebuild macOS custom ops dylib against 2.20.0

## 0.1.5
* Upgrade macOS TFLite native library from 2.11.0 to 2.17.1 (universal binary: arm64 + x86_64)
* Update all C API headers to TFLite 2.17.1 (including new `TfLiteOperator` API replacing `TfLiteRegistrationExternal`)
* Regenerate FFI bindings with new APIs (SignatureRunner, TfLiteInterpreterCancel, and more)
* Rebuild macOS custom ops dylib as universal binary (arm64 + x86_64)

## 0.1.4
* Bundle `libtensorflowlite_c-win.dll` from flutter_litert Windows plugin instead of downstream packages

## 0.1.3
* Fix Windows: build and bundle custom ops DLL (tflite_custom_ops.dll) for MediaPipe models
* Fix heap corruption crash when switching between segmentation models (custom op name string was freed prematurely)

## 0.1.2
* Fix Linux: build and bundle custom ops library (libtflite_custom_ops.so) so MediaPipe models with custom ops (e.g. selfie segmentation) work on Linux

## 0.1.1
* Update AndroidManifest.xml

## 0.1.0
* Fix IsolateInterpreter thread-safety bug causing intermittent native crashes when hardware delegates are active

## 0.0.1
* Initial release, forked from tflite_flutter_custom v1.2.5
* Rebranded to flutter_litert for LiteRT ecosystem
* All native libraries bundled automatically
* Custom ops support (MediaPipe models)
* Full platform support: Android, iOS, macOS, Windows, Linux
