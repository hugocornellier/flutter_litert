// Copyright (c) 2019, the Dart project authors. Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

/// LiteRT (formerly TensorFlow Lite) for Flutter
library;

// Default to the WASM-safe web surface; the native surface (dart:ffi) is gated
// on dart.library.io. Tools that do not evaluate the condition (pana's WASM
// check, `dart analyze`) fall back to the first URI, so it must be the web one.
// Native-only API that cannot be made WASM-safe (it exposes dart:isolate /
// dart:io types) lives in `package:flutter_litert/native.dart` instead.
export 'src/all_web.dart' if (dart.library.io) 'src/all_native.dart';
export 'src/all_common.dart';
