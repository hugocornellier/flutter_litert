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

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import '../bindings/bindings.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';

import '../native/delegate.dart';

/// XNNPack Delegate
class XNNPackDelegate implements Delegate {
  Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  XNNPackDelegate._(this._delegate);

  /// Creates an XNNPACK delegate with optional [options].
  factory XNNPackDelegate({XNNPackDelegateOptions? options}) {
    if (options == null) {
      return XNNPackDelegate._(
        tfliteBinding.TfLiteXNNPackDelegateCreate(nullptr),
      );
    }
    return XNNPackDelegate._(
      tfliteBinding.TfLiteXNNPackDelegateCreate(options.base),
    );
  }

  /// Releases native XNNPACK delegate resources.
  @override
  void delete() {
    checkState(!_deleted, message: 'XNNPackDelegate already deleted.');
    tfliteBinding.TfLiteXNNPackDelegateDelete(_delegate);
    _deleted = true;
  }
}

/// XNNPackDelegate Options
class XNNPackDelegateOptions {
  Pointer<TfLiteXNNPackDelegateOptions> _options;
  bool _deleted = false;
  Pointer<Utf8>? _nativeWeightCacheFilePath;

  /// Pointer to the underlying native options struct.
  Pointer<TfLiteXNNPackDelegateOptions> get base => _options;

  XNNPackDelegateOptions._(this._options, this._nativeWeightCacheFilePath);

  /// Creates XNNPACK delegate options.
  factory XNNPackDelegateOptions({
    int numThreads = 1,
    int flags = 0,
    String? weightCacheFilePath,
  }) {
    final options = calloc<TfLiteXNNPackDelegateOptions>();
    // Build the options struct in Dart rather than calling the native
    // TfLiteXNNPackDelegateOptionsDefault(), which returns the struct by value.
    // That by-value FFI return (a >16-byte struct goes through the AArch64 x8
    // hidden-pointer trampoline) crashes the Dart VM on the arm64 Android
    // emulator; upstream tflite_flutter avoids it the same way. calloc already
    // zero-initialized every field, so we only set what differs from zero and
    // replicate the QS8 + QU8 quantization flags that Default() would have set.
    options.ref.num_threads = numThreads;
    options.ref.flags = flags != 0
        ? flags
        : TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
    // LiteRT 2.3.2 (the Android runtime) appended this field, growing the
    // struct from 56 to 64 bytes on 64-bit hosts, and its delegate copies the
    // whole struct. Older desktop and Apple runtimes stop reading before it.
    options.ref.weight_cache_lock_memory = false;

    Pointer<Utf8>? nativePath;
    if (weightCacheFilePath != null) {
      nativePath = weightCacheFilePath.toNativeUtf8();
      options.ref.weight_cache_file_path = nativePath.cast<Char>();
    }

    return XNNPackDelegateOptions._(options, nativePath);
  }

  /// Releases native resources for these options.
  void delete() {
    checkState(!_deleted, message: 'XNNPackDelegate already deleted.');
    if (_nativeWeightCacheFilePath != null) {
      malloc.free(_nativeWeightCacheFilePath!);
    }
    calloc.free(_options);
    _deleted = true;
  }
}
