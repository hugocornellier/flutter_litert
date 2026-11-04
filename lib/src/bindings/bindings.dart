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

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:flutter_litert/src/bindings/android_runtime_shim.dart';
import 'package:flutter_litert/src/bindings/tensorflow_lite_bindings_generated.dart';
import 'package:flutter_litert/src/delegates/delegate_library_loader.dart';

/// Top-level finals are initialized lazily on first access in Dart.
final DynamicLibrary _dylib = () {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libtensorflowlite_jni.so');
  }

  if (Platform.isIOS) {
    return DynamicLibrary.process();
  }

  // Desktop platforms support multiple loading strategies for testing
  if (Platform.isMacOS || Platform.isLinux || Platform.isWindows) {
    return _loadDesktopLibrary();
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

DynamicLibrary _loadDesktopLibrary() {
  final List<String> attemptedPaths = [];

  final String libName;
  if (Platform.isMacOS) {
    libName = 'libtensorflowlite_c-mac.dylib';
  } else if (Platform.isLinux) {
    libName = 'libtensorflowlite_c-linux.so';
  } else {
    libName = 'libtensorflowlite_c-win.dll';
  }

  final List<String> paths;
  if (Platform.isMacOS) {
    final packagePath = _resolvePackagePath('flutter_litert');
    paths = [
      ...delegateBundlePaths(libName),
      if (packagePath != null)
        '$packagePath/macos/flutter_litert/Sources/flutter_litert/Resources/$libName',
    ];
  } else {
    final execDir = Directory(Platform.resolvedExecutable).parent.path;
    final subDir = Platform.isLinux ? 'lib/' : '';
    final packagePath = _resolvePackagePath('flutter_litert');
    final platformDir = Platform.isLinux ? 'linux/lib' : 'windows';
    paths = [
      '$execDir/$subDir$libName',
      if (packagePath != null) '$packagePath/$platformDir/$libName',
    ];
  }

  final lib = probeLibraryPaths(
    envVar: 'TFLITE_LIB_PATH',
    paths: paths,
    attemptedPaths: attemptedPaths,
  );
  if (lib != null) return lib;

  throw UnsupportedError(
    'Failed to load TensorFlow Lite library. Attempted paths:\n'
    '${attemptedPaths.map((p) => '  - $p').join('\n')}\n\n'
    'Solutions:\n'
    '  1. For production apps: Ensure native libraries are bundled correctly\n'
    '  2. For testing: Set TFLITE_LIB_PATH environment variable:\n'
    '     TFLITE_LIB_PATH=/path/to/library flutter test\n'
    '  3. For testing: Ensure libraries exist in the project fallback locations',
  );
}

/// Resolves the root directory of a Dart package by reading
/// `.dart_tool/package_config.json`. Returns `null` if the package cannot be
/// found. This is used during `flutter test` to locate native libraries
/// bundled inside the flutter_litert package source tree.
String? _resolvePackagePath(String packageName) {
  try {
    final configFile = File(
      '${Directory.current.path}/.dart_tool/package_config.json',
    );
    if (!configFile.existsSync()) return null;

    final config = jsonDecode(configFile.readAsStringSync()) as Map;
    final packages = config['packages'] as List?;
    if (packages == null) return null;

    for (final pkg in packages) {
      if (pkg is Map && pkg['name'] == packageName) {
        final rootUri = pkg['rootUri'] as String?;
        if (rootUri == null) continue;

        if (rootUri.startsWith('file://')) {
          return Uri.parse(rootUri).toFilePath();
        }
        // rootUri is relative to .dart_tool/ directory
        final resolved = configFile.parent.uri.resolve(rootUri);
        return resolved.toFilePath();
      }
    }
  } catch (_) {
    // Silently fail; best-effort fallback
  }
  return null;
}

final DynamicLibrary _dylibGpu = () {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libtensorflowlite_gpu_jni.so');
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// Play services client library that exports the TfLite C API as Gms* symbols.
const String playServicesTfLiteLibrary = 'libtensorflowlite_jni_gms_client.so';

/// Bindings to the TfLite C API of the Google Play services runtime.
///
/// Thrown readiness errors are not cached: Dart re-runs a top-level final's
/// initializer after it throws, so this binds once the runtime is ready.
TensorFlowLiteBindings _playServicesBindings() {
  ensurePlayServicesInterpreterReady();
  final library = DynamicLibrary.open(playServicesTfLiteLibrary);
  return TensorFlowLiteBindings.fromLookup(<T extends NativeType>(name) {
    try {
      return library.lookup<T>('Gms$name');
    } on ArgumentError {
      throw playServicesUnsupported(
        name,
        'Use the default bundled runtime for this feature.',
      );
    }
  });
}

/// TensorFlowLite Bindings
final tfliteBinding = usesPlayServicesRuntime
    ? _playServicesBindings()
    : TensorFlowLiteBindings(_dylib);

/// TensorFlowLite Gpu Bindings
final tfliteBindingGpu = usesPlayServicesRuntime
    ? throw playServicesUnsupported(
        'The GPU delegate',
        'Use CompiledModel with Accelerator.gpu and '
            'initializeAndroidRuntime(enableGpu: true) instead.',
      )
    : TensorFlowLiteBindings(_dylibGpu);
