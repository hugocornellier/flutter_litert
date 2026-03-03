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

import 'package:flutter/services.dart' hide Size;
import 'package:quiver/check.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';
import '../native/delegate.dart';

/// Flex delegate for running models that use `SELECT_TF_OPS`.
///
/// The Flex delegate enables TensorFlow operations that are not available as
/// TFLite builtins. This is required for training models whose gradient ops
/// cannot be expressed as builtins (e.g., Conv2D, BatchNormalization).
///
/// Add [`flutter_litert_flex`](https://pub.dev/packages/flutter_litert_flex)
/// to your `pubspec.yaml` to bundle the native library on all platforms:
///
/// ```yaml
/// dependencies:
///   flutter_litert: ^1.0.3
///   flutter_litert_flex: ^0.0.1
/// ```
///
/// Then use the delegate:
///
/// ```dart
/// final options = InterpreterOptions();
/// options.addDelegate(FlexDelegate());
/// final interpreter = Interpreter.fromFile(model, options: options);
/// ```
class FlexDelegate implements Delegate {
  static DynamicLibrary? _flexLib;

  static Pointer<TfLiteDelegate> Function(
    Pointer<Pointer<Char>>,
    Pointer<Pointer<Char>>,
    int,
    Pointer<NativeFunction<Void Function(Pointer<Char>)>>,
  )?
  _createFn;

  static void Function(Pointer<TfLiteDelegate>)? _destroyFn;

  /// Method channel for Android delegate creation via Java API.
  static const _channel = MethodChannel('flutter_litert_flex');

  Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;
  bool _isAndroid = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  FlexDelegate._(this._delegate, {bool isAndroid = false})
    : _isAndroid = isAndroid;

  /// Creates a [FlexDelegate] for SELECT_TF_OPS support.
  ///
  /// Requires `flutter_litert_flex` in your `pubspec.yaml`.
  ///
  /// On Android this is an async operation internally (method channel),
  /// so use [FlexDelegate.create] for explicit async construction.
  /// This synchronous constructor uses a cached delegate on Android.
  ///
  /// Throws [UnsupportedError] if the library cannot be loaded.
  factory FlexDelegate() {
    if (Platform.isAndroid) {
      if (_androidDelegatePtr == null) {
        throw UnsupportedError(
          'On Android, call FlexDelegate.create() instead of FlexDelegate().\n'
          'The Android FlexDelegate requires async initialization via method channel.',
        );
      }
      final ptr = Pointer<TfLiteDelegate>.fromAddress(_androidDelegatePtr!);
      _androidDelegatePtr = null;
      return FlexDelegate._(ptr, isAndroid: true);
    }
    _loadLibrary();
    final delegate = _createFn!(nullptr, nullptr, 0, nullptr);
    checkArgument(
      delegate != nullptr,
      message: 'Failed to create FlexDelegate (native returned null).',
    );
    return FlexDelegate._(delegate);
  }

  /// Creates a [FlexDelegate] asynchronously.
  ///
  /// This is required on Android where the delegate is created via a method
  /// channel to the Java FlexDelegate API. On other platforms this behaves
  /// identically to the synchronous constructor.
  static Future<FlexDelegate> create() async {
    if (Platform.isAndroid) {
      final handle = await _channel.invokeMethod<int>('createFlexDelegate');
      if (handle == null || handle == 0) {
        throw UnsupportedError(
          'FlexDelegate not available on Android.\n'
          'Add flutter_litert_flex to your pubspec.yaml.',
        );
      }
      return FlexDelegate._(
        Pointer<TfLiteDelegate>.fromAddress(handle),
        isAndroid: true,
      );
    }
    return FlexDelegate();
  }

  static int? _androidDelegatePtr;

  @override
  void delete() {
    checkState(!_deleted, message: 'FlexDelegate already deleted.');
    if (_isAndroid) {
      _channel.invokeMethod<void>('deleteFlexDelegate', _delegate.address);
    } else {
      _destroyFn!(_delegate);
    }
    _deleted = true;
  }

  // ---------------------------------------------------------------------------
  // Static API
  // ---------------------------------------------------------------------------

  /// Whether the Flex delegate library is available.
  ///
  /// Returns `true` if the library can be loaded — i.e., `flutter_litert_flex`
  /// is in the project's dependencies.
  static bool get isAvailable {
    if (_flexLib != null) return true;

    if (Platform.isAndroid) {
      try {
        DynamicLibrary.open('libtensorflowlite_flex_jni.so');
        return true;
      } catch (_) {
        return false;
      }
    }

    // iOS: statically linked at build time by flutter_litert_flex podspec.
    if (Platform.isIOS) {
      try {
        DynamicLibrary.process().lookup<NativeFunction<Void Function()>>(
          'tflite_plugin_create_delegate',
        );
        return true;
      } catch (_) {
        return false;
      }
    }

    // Desktop: check app bundle paths (bundled by flutter_litert_flex).
    return _bundlePaths.any((p) => File(p).existsSync());
  }

  // ---------------------------------------------------------------------------
  // Library loading (non-Android)
  // ---------------------------------------------------------------------------

  /// Paths where the library may exist inside a built app bundle.
  static List<String> get _bundlePaths {
    final libName = _desktopLibName;
    if (Platform.isMacOS) {
      final appBundle = Directory(Platform.resolvedExecutable).parent.parent;
      return [
        '${appBundle.path}/Frameworks/flutter_litert_flex.framework/Versions/A/Resources/$libName',
        '${appBundle.path}/Frameworks/flutter_litert_flex.framework/Resources/$libName',
        '${appBundle.path}/Resources/flutter_litert_flex_flutter_litert_flex.bundle/Contents/Resources/$libName',
        '${appBundle.path}/Resources/$libName',
        '${appBundle.path}/Frameworks/flutter_litert.framework/Versions/A/Resources/$libName',
        '${appBundle.path}/Frameworks/flutter_litert.framework/Resources/$libName',
        '${appBundle.path}/Resources/flutter_litert_flutter_litert.bundle/Contents/Resources/$libName',
      ];
    }
    if (Platform.isLinux) {
      return [
        '${Directory(Platform.resolvedExecutable).parent.path}/lib/$libName',
      ];
    }
    if (Platform.isWindows) {
      return ['${Directory(Platform.resolvedExecutable).parent.path}/$libName'];
    }
    return [];
  }

  static String get _desktopLibName {
    if (Platform.isMacOS) return 'libtensorflowlite_flex-mac.dylib';
    if (Platform.isLinux) return 'libtensorflowlite_flex-linux.so';
    if (Platform.isWindows) return 'libtensorflowlite_flex-win.dll';
    throw UnsupportedError(
      'FlexDelegate desktop lib is not supported on ${Platform.operatingSystem}',
    );
  }

  static void _loadLibrary() {
    if (_flexLib != null) return;

    _flexLib = _openLibrary();

    _createFn = _flexLib!
        .lookupFunction<
          Pointer<TfLiteDelegate> Function(
            Pointer<Pointer<Char>>,
            Pointer<Pointer<Char>>,
            Size,
            Pointer<NativeFunction<Void Function(Pointer<Char>)>>,
          ),
          Pointer<TfLiteDelegate> Function(
            Pointer<Pointer<Char>>,
            Pointer<Pointer<Char>>,
            int,
            Pointer<NativeFunction<Void Function(Pointer<Char>)>>,
          )
        >('tflite_plugin_create_delegate');

    _destroyFn = _flexLib!
        .lookupFunction<
          Void Function(Pointer<TfLiteDelegate>),
          void Function(Pointer<TfLiteDelegate>)
        >('tflite_plugin_destroy_delegate');
  }

  static DynamicLibrary _openLibrary() {
    // iOS: symbols are statically linked into the app binary.
    if (Platform.isIOS) {
      try {
        return DynamicLibrary.process();
      } catch (e) {
        throw UnsupportedError(
          'FlexDelegate not available on iOS.\n'
          'Add flutter_litert_flex to your pubspec.yaml.',
        );
      }
    }

    // Desktop: try app bundle paths (bundled by flutter_litert_flex).
    final List<String> attemptedPaths = [];
    for (final path in _bundlePaths) {
      attemptedPaths.add(path);
      try {
        return DynamicLibrary.open(path);
      } catch (e) {
        // Continue
      }
    }

    throw UnsupportedError(
      'FlexDelegate library not found.\n'
      'Add flutter_litert_flex to your pubspec.yaml.\n\n'
      'Attempted paths:\n'
      '${attemptedPaths.map((p) => '  - $p').join('\n')}',
    );
  }
}
