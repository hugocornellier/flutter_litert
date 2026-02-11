/*
 * Copyright 2025 flutter_litert authors.
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
import 'package:flutter_litert/src/bindings/bindings.dart';
import 'package:flutter_litert/src/bindings/tensorflow_lite_bindings_generated.dart';

/// Loads and provides access to the Convolution2DTransposeBias custom op.
///
/// This custom op is required for MediaPipe models like Selfie Segmentation.
class TransposeConvBiasOp {
  static DynamicLibrary? _customOpsLib;
  static Pointer<TfLiteRegistration>? _registration;
  static bool _isRegistered = false;

  /// Returns whether the custom op has been successfully loaded.
  static bool get isLoaded => _registration != null;

  /// Returns whether the custom op has been registered with an interpreter options.
  static bool get isRegistered => _isRegistered;

  /// Loads the custom ops library.
  ///
  /// This is called automatically when needed, but can be called early
  /// to catch loading errors.
  static void loadLibrary() {
    if (_customOpsLib != null) return;

    _customOpsLib = _loadCustomOpsLibrary();
    if (_customOpsLib == null) {
      throw UnsupportedError('Failed to load custom ops library');
    }

    // Get the registration function
    final registerFn = _customOpsLib!.lookupFunction<
        Pointer<TfLiteRegistration> Function(),
        Pointer<TfLiteRegistration> Function()>(
      'TfLiteFlutter_RegisterConvolution2DTransposeBias',
    );

    _registration = registerFn();
  }

  /// Registers the Convolution2DTransposeBias custom op with the given interpreter options.
  ///
  /// Call this before creating an interpreter for models that use this op.
  static void registerWithOptions(Pointer<TfLiteInterpreterOptions> options) {
    if (_registration == null) {
      loadLibrary();
    }

    if (_registration == null) {
      throw StateError('Custom op registration not available');
    }

    final opName = 'Convolution2DTransposeBias'.toNativeUtf8().cast<Char>();
    tfliteBinding.TfLiteInterpreterOptionsAddCustomOp(
      options,
      opName,
      _registration!,
      1, // min_version
      1, // max_version
    );
    calloc.free(opName);

    _isRegistered = true;
  }

  /// Attempts to load the custom ops library from various locations.
  static DynamicLibrary? _loadCustomOpsLibrary() {
    // iOS: Custom ops are statically linked into the app via CocoaPods
    // Use DynamicLibrary.process() to access symbols from the main executable
    if (Platform.isIOS) {
      try {
        return DynamicLibrary.process();
      } catch (e) {
        // Fall back to DynamicLibrary.executable() if process() fails
        try {
          return DynamicLibrary.executable();
        } catch (e) {
          return null;
        }
      }
    }

    // Android: Custom ops are built as a separate .so via CMake
    if (Platform.isAndroid) {
      try {
        return DynamicLibrary.open('libtflite_custom_ops.so');
      } catch (e) {
        return null;
      }
    }

    final List<String> attemptedPaths = [];

    // Desktop platforms: Check for environment variable override
    final envPath = Platform.environment['TFLITE_CUSTOM_OPS_PATH'];
    if (envPath != null && envPath.isNotEmpty) {
      attemptedPaths.add('TFLITE_CUSTOM_OPS_PATH: $envPath');
      try {
        return DynamicLibrary.open(envPath);
      } catch (e) {
        // Continue to fallback paths
      }
    }

    String libName;
    if (Platform.isMacOS) {
      libName = 'libtflite_custom_ops.dylib';
    } else if (Platform.isLinux) {
      libName = 'libtflite_custom_ops.so';
    } else if (Platform.isWindows) {
      libName = 'tflite_custom_ops.dll';
    } else {
      // Unknown platform
      return null;
    }

    // Desktop: Try production app bundle path
    String productionPath;
    if (Platform.isMacOS) {
      productionPath =
          '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/$libName';
    } else if (Platform.isLinux) {
      productionPath =
          '${Directory(Platform.resolvedExecutable).parent.path}/lib/$libName';
    } else {
      productionPath =
          '${Directory(Platform.resolvedExecutable).parent.path}/$libName';
    }

    attemptedPaths.add('Production path: $productionPath');
    try {
      return DynamicLibrary.open(productionPath);
    } catch (e) {
      // Continue to fallback paths
    }

    // macOS: Check various locations where CocoaPods puts libraries
    if (Platform.isMacOS) {
      final appBundle = Directory(Platform.resolvedExecutable).parent.parent;

      // Check inside flutter_litert.framework/Resources
      // This is where CocoaPods puts s.resources for framework targets
      final frameworkResourcesPath =
          '${appBundle.path}/Frameworks/flutter_litert.framework/Versions/A/Resources/$libName';
      attemptedPaths.add('Framework Resources path: $frameworkResourcesPath');
      try {
        return DynamicLibrary.open(frameworkResourcesPath);
      } catch (e) {
        // Continue
      }

      // Also check without Versions/A (for symlinked frameworks)
      final frameworkResourcesPathAlt =
          '${appBundle.path}/Frameworks/flutter_litert.framework/Resources/$libName';
      attemptedPaths
          .add('Framework Resources path (alt): $frameworkResourcesPathAlt');
      try {
        return DynamicLibrary.open(frameworkResourcesPathAlt);
      } catch (e) {
        // Continue
      }

      // App's Resources directory (fallback)
      final resourcesPath = '${appBundle.path}/Resources/$libName';
      attemptedPaths.add('Resources path: $resourcesPath');
      try {
        return DynamicLibrary.open(resourcesPath);
      } catch (e) {
        // Continue
      }

      // Frameworks directory (fallback)
      final frameworksPath = '${appBundle.path}/Frameworks/$libName';
      attemptedPaths.add('Frameworks path: $frameworksPath');
      try {
        return DynamicLibrary.open(frameworksPath);
      } catch (e) {
        // Continue
      }

      final fallbackPath = '${Directory.current.path}/macos/$libName';
      attemptedPaths.add('Fallback path: $fallbackPath');
      try {
        return DynamicLibrary.open(fallbackPath);
      } catch (e) {
        // Continue
      }
    }

    // If we got here, library loading failed
    return null;
  }
}
