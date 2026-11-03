/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
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

/// Android library built from src/play_services/flutter_litert_runtime.c.
///
/// It records which LiteRT runtime the app was built with and, for the Google
/// Play services runtime, the initialization state of each half of it. That
/// state is process-wide, so helper isolates see what the main isolate's
/// initializeAndroidRuntime() call did.
const String androidRuntimeShimLibrary = 'libflutter_litert_runtime.so';

/// Values of FlutterLiteRtRuntime*State() in flutter_litert_runtime.c.
const int kRuntimeStateNotInitialized = 0;
const int kRuntimeStateReady = 1;
const int kRuntimeStateFailed = 2;

const int _kRuntimeModePlayServices = 1;

typedef _Int32Native = Int32 Function();
typedef _Int32Dart = int Function();
typedef _StringNative = Pointer<Utf8> Function();
typedef _PointerNative = Pointer<Void> Function();

final class AndroidRuntimeShim {
  AndroidRuntimeShim._(DynamicLibrary library)
    : _mode = library.lookupFunction<_Int32Native, _Int32Dart>(
        'FlutterLiteRtRuntimeMode',
      ),
      _interpreterState = library.lookupFunction<_Int32Native, _Int32Dart>(
        'FlutterLiteRtRuntimeInterpreterState',
      ),
      _compiledModelState = library.lookupFunction<_Int32Native, _Int32Dart>(
        'FlutterLiteRtRuntimeCompiledModelState',
      ),
      _interpreterError = library.lookupFunction<_StringNative, _StringNative>(
        'FlutterLiteRtRuntimeInterpreterError',
      ),
      _compiledModelError = library
          .lookupFunction<_StringNative, _StringNative>(
            'FlutterLiteRtRuntimeCompiledModelError',
          ),
      _api = library.lookupFunction<_PointerNative, _PointerNative>(
        'FlutterLiteRtRuntimeApi',
      ),
      _gpuAccelerator = library.lookupFunction<_PointerNative, _PointerNative>(
        'FlutterLiteRtRuntimeGpuAccelerator',
      ),
      _context = library.lookupFunction<_PointerNative, _PointerNative>(
        'FlutterLiteRtRuntimeContext',
      ),
      library = library;

  /// The opened shim. In the Play services mode it also exports the LiteRt C
  /// API, forwarded to the Play services runtime.
  final DynamicLibrary library;
  final _Int32Dart _mode;
  final _Int32Dart _interpreterState;
  final _Int32Dart _compiledModelState;
  final _StringNative _interpreterError;
  final _StringNative _compiledModelError;
  final _PointerNative _api;
  final _PointerNative _gpuAccelerator;
  final _PointerNative _context;

  bool get isPlayServices => _mode() == _kRuntimeModePlayServices;

  int get interpreterState => _interpreterState();
  int get compiledModelState => _compiledModelState();
  String get interpreterError => _interpreterError().toDartString();
  String get compiledModelError => _compiledModelError().toDartString();

  /// The Play services LiteRtRuntimeCApiStruct, or nullptr before it is ready.
  Pointer<Void> get runtimeApi => _api();

  /// The Play services GPU accelerator, or nullptr when it has none.
  Pointer<Void> get gpuAccelerator => _gpuAccelerator();

  /// The Play services runtime context, or nullptr when it has none.
  Pointer<Void> get context => _context();
}

/// The Android runtime shim, or null off Android.
///
/// An APK without the shim cannot come from this plugin version, but treat it
/// as the bundled runtime rather than failing, since that is what such an APK
/// would contain.
final AndroidRuntimeShim? androidRuntimeShim = () {
  if (!Platform.isAndroid) return null;
  try {
    return AndroidRuntimeShim._(DynamicLibrary.open(androidRuntimeShimLibrary));
  } on ArgumentError {
    return null;
  }
}();

/// Whether the app was built with flutterLitert.runtime=playServices.
final bool usesPlayServicesRuntime =
    androidRuntimeShim?.isPlayServices ?? false;

/// Throws unless the Play services runtime half named [component] is ready.
///
/// [state] and [error] are read from [androidRuntimeShim].
void checkPlayServicesRuntimeReady({
  required String component,
  required int state,
  required String Function() error,
}) {
  if (state == kRuntimeStateReady) return;
  if (state == kRuntimeStateFailed) {
    throw StateError(
      'flutter_litert: the Google Play services LiteRT $component runtime is '
      'unavailable: ${error()}. Call initializeAndroidRuntime() again to '
      'retry.',
    );
  }
  throw StateError(
    'flutter_litert: this app is built with flutterLitert.runtime=playServices, '
    'so LiteRT comes from Google Play services and must be initialized before '
    'an $component is created. Call `await initializeAndroidRuntime()` from '
    'the main isolate first.',
  );
}

/// Throws a [StateError] unless the Play services Interpreter runtime is ready.
void ensurePlayServicesInterpreterReady() {
  final shim = androidRuntimeShim!;
  checkPlayServicesRuntimeReady(
    component: 'Interpreter',
    state: shim.interpreterState,
    error: () => shim.interpreterError,
  );
}

/// Throws a [StateError] unless the Play services CompiledModel runtime is
/// ready.
void ensurePlayServicesCompiledModelReady() {
  final shim = androidRuntimeShim!;
  checkPlayServicesRuntimeReady(
    component: 'CompiledModel',
    state: shim.compiledModelState,
    error: () => shim.compiledModelError,
  );
}

/// Error for a feature that the Play services runtime does not provide.
UnsupportedError playServicesUnsupported(String feature, [String? hint]) {
  return UnsupportedError(
    'flutter_litert: $feature is not available with the Google Play services '
    'LiteRT runtime (flutterLitert.runtime=playServices).'
    '${hint == null ? '' : ' $hint'}',
  );
}
