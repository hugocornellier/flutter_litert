/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import '../bindings/bindings.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';
import '../native/delegate.dart';
import 'delegate_library_loader.dart';

// Re-export the Metal wait-type enum so callers using the public barrel can
// name it without importing the generated bindings.
export '../bindings/tensorflow_lite_bindings_generated.dart'
    show TFLGpuDelegateWaitType;

/// Lazily loaded Metal-specific binding.
///
/// On iOS the Metal symbols live in the main process (statically linked from
/// TensorFlowLiteCMetal.xcframework), so we reuse [tfliteBinding].
///
/// On macOS the core TFLite dylib has no Metal symbols. A separate
/// `libtensorflowlite_gpu-mac.dylib` is bundled in the app resources.
final TensorFlowLiteBindings _metalBinding = () {
  if (Platform.isIOS) return tfliteBinding;
  if (Platform.isMacOS) return TensorFlowLiteBindings(_openMetalLibrary());
  throw UnsupportedError(
    'Metal GPU delegate is not supported on ${Platform.operatingSystem}',
  );
}();

/// Metal Delegate for iOS and macOS
/// Prefer [PerformanceConfig] over constructing this directly: it picks a
/// sensible delegate per platform and handles the fallback when one declines a
/// model. This class stays available for callers that need to configure a
/// delegate themselves.
///
/// Not deprecated in favour of `CompiledModel`. LiteRT Next is the direction
/// upstream is taking, but as of flutter_litert 3.7.0 it miscomputes models
/// whose output tensor ends up dynamic, returning `kLiteRtStatusOk` while the
/// output buffer is never written (see doc/delegate_verification.md and
/// [verifyCompiledModel]). Until that is fixed upstream, delegate-backed
/// `Interpreter` inference is the correct choice for those models, so pointing
/// callers away from it would be wrong.
class GpuDelegate implements Delegate {
  static DynamicLibrary? _metalLib;

  Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  GpuDelegate._(this._delegate);

  /// Creates a Metal GPU delegate with optional [options].
  factory GpuDelegate({GpuDelegateOptions? options}) {
    if (options == null) {
      return GpuDelegate._(_metalBinding.TFLGpuDelegateCreate(nullptr));
    }
    return GpuDelegate._(_metalBinding.TFLGpuDelegateCreate(options.base));
  }

  /// Releases native Metal delegate resources.
  @override
  void delete() {
    checkState(!_deleted, message: 'TfLiteGpuDelegate already deleted.');
    _metalBinding.TFLGpuDelegateDelete(_delegate);
    _deleted = true;
  }

  /// Binds a Metal buffer to an input or output tensor.
  ///
  /// The bound buffer must have sufficient storage for all tensor elements.
  /// For quantized models, the buffer is bound to the internal dequantized
  /// float32 tensor.
  ///
  /// Must be called *after* the delegate has been applied to the interpreter.
  /// Returns true on success.
  bool bindMetalBufferToTensor(int tensorIndex, int metalBuffer) {
    checkState(!_deleted, message: 'TfLiteGpuDelegate already deleted.');
    return _metalBinding.TFLGpuDelegateBindMetalBufferToTensor(
      _delegate,
      tensorIndex,
      metalBuffer,
    );
  }

  // ---------------------------------------------------------------------------
  // Library loading (private)
  // ---------------------------------------------------------------------------

  static String get _libName => 'libtensorflowlite_gpu-mac.dylib';

  /// Paths where the library may exist inside a built app bundle.
  static List<String> get _bundlePaths => delegateBundlePaths(_libName);
}

/// Metal Delegate options
/// Options for a delegate that is not deprecated; see the delegate class for
/// why, and prefer [PerformanceConfig] unless you need this level of control.
class GpuDelegateOptions {
  Pointer<TFLGpuDelegateOptions> _options;
  bool _deleted = false;

  /// Pointer to the underlying native options struct.
  Pointer<TFLGpuDelegateOptions> get base => _options;

  GpuDelegateOptions._(this._options);

  /// Creates default Metal delegate options.
  factory GpuDelegateOptions({
    bool allowPrecisionLoss = false,
    int waitType = TFLGpuDelegateWaitType.TFLGpuDelegateWaitTypePassive,
    bool enableQuantization = true,
  }) {
    final options = calloc<TFLGpuDelegateOptions>();
    options.ref = _metalBinding.TFLGpuDelegateOptionsDefault();
    options.ref
      ..allow_precision_loss = allowPrecisionLoss
      ..wait_type = waitType
      ..enable_quantization = enableQuantization;

    return GpuDelegateOptions._(options);
  }

  /// Releases native resources for these options.
  void delete() {
    checkState(!_deleted, message: 'TfLiteGpuDelegate already deleted.');
    calloc.free(_options);
    _deleted = true;
  }
}

/// Opens the Metal GPU delegate dylib on macOS.
DynamicLibrary _openMetalLibrary() => openDelegateLibrary(
  envVar: 'TFLITE_METAL_PATH',
  bundlePaths: GpuDelegate._bundlePaths,
  description: 'Metal GPU delegate',
  getCached: () => GpuDelegate._metalLib,
  setCached: (lib) => GpuDelegate._metalLib = lib,
);
