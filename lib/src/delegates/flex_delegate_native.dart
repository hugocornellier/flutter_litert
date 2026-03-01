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

import 'package:quiver/check.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';
import '../native/delegate.dart';

/// Flex delegate for running models that use `SELECT_TF_OPS`.
///
/// The Flex delegate enables TensorFlow operations that are not available as
/// TFLite builtins. This is required for training models whose gradient ops
/// cannot be expressed as builtins (e.g., Conv2D, BatchNormalization).
///
/// Add the [`flutter_litert_flex`](https://pub.dev/packages/flutter_litert_flex)
/// package to your `pubspec.yaml` to bundle the native library automatically
/// on all platforms:
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
///
/// Alternatively, on desktop call [download] to fetch the library manually,
/// or on Android add the Maven dependency directly.
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

  Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  FlexDelegate._(this._delegate);

  /// Creates a [FlexDelegate] for SELECT_TF_OPS support.
  ///
  /// The flex native library must be available before calling this constructor.
  /// On desktop, call [download] first to ensure it is cached.
  /// On Android, add the tensorflow-lite-select-tf-ops Maven dependency.
  ///
  /// Throws [UnsupportedError] if the library cannot be loaded.
  factory FlexDelegate() {
    _loadLibrary();
    final delegate = _createFn!(nullptr, nullptr, 0, nullptr);
    checkArgument(
      delegate != nullptr,
      message: 'Failed to create FlexDelegate (native returned null).',
    );
    return FlexDelegate._(delegate);
  }

  @override
  void delete() {
    checkState(!_deleted, message: 'FlexDelegate already deleted.');
    _destroyFn!(_delegate);
    _deleted = true;
  }

  // ---------------------------------------------------------------------------
  // Static API
  // ---------------------------------------------------------------------------

  /// Whether the Flex delegate library is available locally.
  ///
  /// Returns `true` if the library can be loaded right now (without
  /// triggering a download). Checks the environment variable, user cache,
  /// and app bundle paths. On Android this attempts a system library load.
  static bool get isAvailable {
    if (_flexLib != null) return true;

    if (Platform.isAndroid) {
      try {
        DynamicLibrary.open(_libName);
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

    final envPath = Platform.environment['TFLITE_FLEX_PATH'];
    if (envPath != null && envPath.isNotEmpty && File(envPath).existsSync()) {
      return true;
    }

    if (File('${_cacheDir.path}/$_libName').existsSync()) {
      return true;
    }

    // Check app bundle paths (library may be auto-bundled at build time).
    return _bundlePaths.any((p) => File(p).existsSync());
  }

  /// Downloads the Flex delegate native library from GitHub Releases.
  ///
  /// The library is cached locally. Subsequent calls are no-ops if the
  /// library already exists. This is a no-op on Android where the library
  /// comes from the Maven dependency.
  static Future<void> download({String version = '1.0.0'}) async {
    if (Platform.isAndroid) return;
    if (isAvailable) return;

    final dir = _cacheDir;
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }

    final libName = _libName;
    final url = Uri.parse(
      'https://github.com/hugocornellier/flutter_litert/releases/download/'
      'flex-v$version/$libName',
    );

    final client = HttpClient();
    try {
      final request = await client.getUrl(url);
      final response = await request.close();

      if (response.statusCode != 200) {
        throw StateError(
          'Failed to download FlexDelegate library: '
          'HTTP ${response.statusCode} from $url',
        );
      }

      // Write to a temp file and rename for atomicity.
      final tmpFile = File('${dir.path}/$libName.tmp');
      final sink = tmpFile.openWrite();
      await response.pipe(sink);
      await tmpFile.rename('${dir.path}/$libName');
    } finally {
      client.close();
    }
  }

  // ---------------------------------------------------------------------------
  // Library loading
  // ---------------------------------------------------------------------------

  static String get _libName {
    if (Platform.isMacOS) return 'libtensorflowlite_flex-mac.dylib';
    if (Platform.isLinux) return 'libtensorflowlite_flex-linux.so';
    if (Platform.isWindows) return 'libtensorflowlite_flex-win.dll';
    if (Platform.isAndroid) return 'libtensorflowlite_flex.so';
    throw UnsupportedError(
      'FlexDelegate is not supported on ${Platform.operatingSystem}',
    );
  }

  static Directory get _cacheDir {
    if (Platform.isMacOS) {
      return Directory(
        '${Platform.environment['HOME']}/Library/Caches/flutter_litert',
      );
    }
    if (Platform.isLinux) {
      final xdgCache = Platform.environment['XDG_CACHE_HOME'];
      final base = xdgCache ?? '${Platform.environment['HOME']}/.cache';
      return Directory('$base/flutter_litert');
    }
    if (Platform.isWindows) {
      final localAppData = Platform.environment['LOCALAPPDATA']!;
      return Directory('$localAppData\\flutter_litert\\cache');
    }
    throw UnsupportedError(
      'FlexDelegate cache is not supported on ${Platform.operatingSystem}',
    );
  }

  /// Paths where the library may exist inside a built app bundle.
  static List<String> get _bundlePaths {
    final libName = _libName;
    if (Platform.isMacOS) {
      final appBundle = Directory(Platform.resolvedExecutable).parent.parent;
      return [
        '${appBundle.path}/Resources/$libName',
        '${appBundle.path}/Frameworks/flutter_litert.framework/Versions/A/Resources/$libName',
        '${appBundle.path}/Frameworks/flutter_litert.framework/Resources/$libName',
        '${appBundle.path}/Resources/flutter_litert_flutter_litert.bundle/Contents/Resources/$libName',
        // flutter_litert_flex bundle path
        '${appBundle.path}/Resources/flutter_litert_flex_flutter_litert_flex.bundle/Contents/Resources/$libName',
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
    final List<String> attemptedPaths = [];

    // iOS: symbols are statically linked into the app binary.
    if (Platform.isIOS) {
      try {
        return DynamicLibrary.process();
      } catch (e) {
        throw UnsupportedError(
          'FlexDelegate not available on iOS.\n'
          'Add flutter_litert_flex to your pubspec.yaml to bundle it.',
        );
      }
    }

    // Android: load from system (Maven dependency)
    if (Platform.isAndroid) {
      try {
        return DynamicLibrary.open(_libName);
      } catch (e) {
        throw UnsupportedError(
          'FlexDelegate library not available on Android.\n'
          'Add to android/app/build.gradle:\n'
          "  implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:+'",
        );
      }
    }

    // Desktop: check environment variable override
    final envPath = Platform.environment['TFLITE_FLEX_PATH'];
    if (envPath != null && envPath.isNotEmpty) {
      attemptedPaths.add('TFLITE_FLEX_PATH: $envPath');
      try {
        return DynamicLibrary.open(envPath);
      } catch (e) {
        // Continue to fallback paths
      }
    }

    final libName = _libName;

    // Desktop: check cache directory
    final cachedPath = '${_cacheDir.path}/$libName';
    attemptedPaths.add('Cache path: $cachedPath');
    try {
      return DynamicLibrary.open(cachedPath);
    } catch (e) {
      // Continue
    }

    // Desktop: try production app bundle paths (auto-bundled at build time)
    for (final path in _bundlePaths) {
      attemptedPaths.add(path);
      try {
        return DynamicLibrary.open(path);
      } catch (e) {
        // Continue
      }
    }

    throw UnsupportedError(
      'FlexDelegate library not found. Attempted paths:\n'
      '${attemptedPaths.map((p) => '  - $p').join('\n')}\n\n'
      'Solutions:\n'
      '  1. Add flutter_litert_flex to your pubspec.yaml (recommended)\n'
      '  2. Call await FlexDelegate.download() first\n'
      '  3. Set TFLITE_FLEX_PATH environment variable to the library path\n',
    );
  }
}
