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
class CoreMlDelegate implements Delegate {
  static DynamicLibrary? _coremlLib;

  Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  CoreMlDelegate._(this._delegate);

  factory CoreMlDelegate({CoreMlDelegateOptions? options}) {
    final delegateOptions = options ?? CoreMlDelegateOptions();

    return CoreMlDelegate._(
      _coremlBinding.TfLiteCoreMlDelegateCreate(delegateOptions.base),
    );
  }

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
  static List<String> get _bundlePaths {
    final libName = _libName;
    final appBundle = Directory(Platform.resolvedExecutable).parent.parent;
    return [
      '${appBundle.path}/Resources/$libName',
      '${appBundle.path}/Frameworks/flutter_litert.framework/Versions/A/Resources/$libName',
      '${appBundle.path}/Frameworks/flutter_litert.framework/Resources/$libName',
      '${appBundle.path}/Resources/flutter_litert_flutter_litert.bundle/Contents/Resources/$libName',
    ];
  }
}

/// CoreMlDelegate Options
class CoreMlDelegateOptions {
  Pointer<TfLiteCoreMlDelegateOptions> _options;
  bool _deleted = false;

  Pointer<TfLiteCoreMlDelegateOptions> get base => _options;

  CoreMlDelegateOptions._(this._options);

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

  void delete() {
    checkState(!_deleted, message: 'CoreMlDelegate already deleted.');
    calloc.free(_options);
    _deleted = true;
  }
}

/// Opens the CoreML delegate dylib on macOS.
DynamicLibrary _openCoremlLibrary() {
  if (CoreMlDelegate._coremlLib != null) return CoreMlDelegate._coremlLib!;

  final List<String> attemptedPaths = [];

  // Strategy 1: Environment variable override
  final envPath = Platform.environment['TFLITE_COREML_PATH'];
  if (envPath != null && envPath.isNotEmpty) {
    attemptedPaths.add('TFLITE_COREML_PATH: $envPath');
    try {
      final lib = DynamicLibrary.open(envPath);
      CoreMlDelegate._coremlLib = lib;
      return lib;
    } catch (e) {
      // Continue
    }
  }

  // Strategy 2: App bundle paths (dylib is bundled in the package)
  for (final path in CoreMlDelegate._bundlePaths) {
    attemptedPaths.add(path);
    try {
      final lib = DynamicLibrary.open(path);
      CoreMlDelegate._coremlLib = lib;
      return lib;
    } catch (e) {
      // Continue
    }
  }

  throw UnsupportedError(
    'CoreML delegate library not found. Attempted paths:\n'
    '${attemptedPaths.map((p) => '  - $p').join('\n')}\n\n'
    'Solutions:\n'
    '  1. Set TFLITE_COREML_PATH environment variable to the library path\n'
    '  2. The CoreML delegate requires macOS on Apple Silicon (arm64)\n',
  );
}
