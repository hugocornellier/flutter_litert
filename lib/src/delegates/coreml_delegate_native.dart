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

// Re-export the CoreML device-selection enum so callers using the public
// barrel can name it without importing the generated bindings.
export '../bindings/tensorflow_lite_bindings_generated.dart'
    show TfLiteCoreMlDelegateEnabledDevices;

/// Lazily loaded CoreML-specific binding.
///
/// On iOS the CoreML symbols live in the main process (statically linked from
/// TensorFlowLiteCCoreML.xcframework), so we reuse [tfliteBinding].
///
/// On macOS the core TFLite dylib has no CoreML symbols. A separate
/// `libtensorflowlite_coreml-mac.dylib` is bundled in the app resources.
final TensorFlowLiteBindings _coremlBinding = () {
  if (Platform.isIOS) return tfliteBinding;
  if (Platform.isMacOS) return TensorFlowLiteBindings(_openCoremlLibrary());
  throw UnsupportedError(
    'CoreML delegate is not supported on ${Platform.operatingSystem}',
  );
}();

/// CoreMl Delegate
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
class CoreMlDelegate implements Delegate {
  static DynamicLibrary? _coremlLib;

  Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  CoreMlDelegate._(this._delegate);

  /// Creates a CoreML delegate with optional [options].
  ///
  /// When [options] is omitted a default set is allocated internally and freed
  /// again before returning: `TfLiteCoreMlDelegateCreate` copies what it needs
  /// out of the struct, so keeping it alive past the call served no purpose and
  /// leaked it for the life of the process. Caller-supplied [options] are left
  /// alone, since the caller owns them and may reuse them for another delegate.
  factory CoreMlDelegate({CoreMlDelegateOptions? options}) {
    final delegateOptions = options ?? CoreMlDelegateOptions();
    try {
      return CoreMlDelegate._(
        _coremlBinding.TfLiteCoreMlDelegateCreate(delegateOptions.base),
      );
    } finally {
      if (options == null) delegateOptions.delete();
    }
  }

  /// Releases native CoreML delegate resources.
  @override
  void delete() {
    checkState(!_deleted, message: 'CoreMlDelegate already deleted.');
    _coremlBinding.TfLiteCoreMlDelegateDelete(_delegate);
    _deleted = true;
  }

  // ---------------------------------------------------------------------------
  // Library loading (private)
  // ---------------------------------------------------------------------------

  static String get _libName => 'libtensorflowlite_coreml-mac.dylib';

  /// Paths where the library may exist inside a built app bundle.
  static List<String> get _bundlePaths => delegateBundlePaths(_libName);
}

/// CoreMlDelegate Options
/// Options for a delegate that is not deprecated; see the delegate class for
/// why, and prefer [PerformanceConfig] unless you need this level of control.
class CoreMlDelegateOptions {
  Pointer<TfLiteCoreMlDelegateOptions> _options;
  bool _deleted = false;

  /// Pointer to the underlying native options struct.
  Pointer<TfLiteCoreMlDelegateOptions> get base => _options;

  CoreMlDelegateOptions._(this._options);

  /// Creates CoreML delegate options.
  factory CoreMlDelegateOptions({
    int enabledDevices = TfLiteCoreMlDelegateEnabledDevices
        .TfLiteCoreMlDelegateDevicesWithNeuralEngine,
    int coremlVersion = 0,
    int maxDelegatedPartitions = 0,
    int minNodesPerPartition = 2,
  }) {
    final options = calloc<TfLiteCoreMlDelegateOptions>();

    options.ref
      ..enabled_devices = enabledDevices
      ..coreml_version = coremlVersion
      ..max_delegated_partitions = maxDelegatedPartitions
      ..min_nodes_per_partition = minNodesPerPartition;

    return CoreMlDelegateOptions._(options);
  }

  /// Releases native resources for these options.
  void delete() {
    checkState(!_deleted, message: 'CoreMlDelegate already deleted.');
    calloc.free(_options);
    _deleted = true;
  }
}

/// Opens the CoreML delegate dylib on macOS.
DynamicLibrary _openCoremlLibrary() => openDelegateLibrary(
  envVar: 'TFLITE_COREML_PATH',
  bundlePaths: CoreMlDelegate._bundlePaths,
  description: 'CoreML delegate',
  getCached: () => CoreMlDelegate._coremlLib,
  setCached: (lib) => CoreMlDelegate._coremlLib = lib,
);
