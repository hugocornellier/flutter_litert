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
import 'delegate.dart';
import '../custom_ops/transpose_conv_bias.dart';

final class _CustomOpConfiguration {
  const _CustomOpConfiguration({
    required this.name,
    required this.registration,
    required this.minVersion,
    required this.maxVersion,
  });

  final String name;
  final Pointer<TfLiteRegistration> registration;
  final int minVersion;
  final int maxVersion;
}

/// LiteRT interpreter options.
class InterpreterOptions {
  final Pointer<TfLiteInterpreterOptions> _options;
  final List<Pointer<Char>> _customOpNames = [];
  final List<_CustomOpConfiguration> _customOps = [];
  bool _deleted = false;
  bool _hasDelegate = false;
  bool _hasMediaPipeCustomOps = false;
  int? _threads;

  Pointer<TfLiteInterpreterOptions> get base => _options;

  /// Whether a delegate has been added via [addDelegate].
  ///
  /// Lets interpreter creation fall back to CPU if the delegate cannot be
  /// applied to a model on the current runtime.
  bool get hasDelegate => _hasDelegate;

  /// Configured CPU thread count, or null when the runtime default is used.
  int? get threads => _threads;

  InterpreterOptions._(this._options);

  /// Creates a new options instance.
  factory InterpreterOptions() =>
      InterpreterOptions._(tfliteBinding.TfLiteInterpreterOptionsCreate());

  /// Destroys the options instance.
  void delete() {
    checkState(!_deleted, message: 'InterpreterOptions already deleted.');
    tfliteBinding.TfLiteInterpreterOptionsDelete(_options);
    for (final name in _customOpNames) {
      calloc.free(name);
    }
    _customOpNames.clear();
    _deleted = true;
  }

  /// Sets the number of CPU threads to use.
  set threads(int threads) {
    checkState(!_deleted, message: 'InterpreterOptions already deleted.');
    _threads = threads;
    tfliteBinding.TfLiteInterpreterOptionsSetNumThreads(_options, threads);
  }

  /// Adds delegate to Interpreter Options
  void addDelegate(Delegate delegate) {
    tfliteBinding.TfLiteInterpreterOptionsAddDelegate(_options, delegate.base);
    _hasDelegate = true;
  }

  /// Registers a custom op with these interpreter options.
  ///
  /// [registration] must point to a valid `TfLiteRegistration` returned by your
  /// native custom-op library. The registration's contents must remain valid
  /// for the lifetime of every interpreter created from these options.
  ///
  /// The op [name] must match the custom op name embedded in the `.tflite`
  /// model. This method keeps the native op-name string alive until [delete].
  ///
  /// Call this before creating the interpreter.
  ///
  /// Example:
  /// ```dart
  /// final options = InterpreterOptions();
  /// options.addCustomOp(
  ///   name: 'MyCustomOpName',
  ///   registration: registrationPointer,
  /// );
  /// final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
  /// ```
  void addCustomOp<T extends NativeType>({
    required String name,
    required Pointer<T> registration,
    int minVersion = 1,
    int maxVersion = 1,
  }) {
    checkState(!_deleted, message: 'InterpreterOptions already deleted.');
    checkArgument(
      name.isNotEmpty,
      message: 'Custom op name must not be empty.',
    );
    checkArgument(
      minVersion >= 1,
      message: 'minVersion must be greater than or equal to 1.',
    );
    checkArgument(
      maxVersion >= minVersion,
      message: 'maxVersion must be greater than or equal to minVersion.',
    );

    final opName = name.toNativeUtf8().cast<Char>();
    _customOpNames.add(opName);
    _customOps.add(
      _CustomOpConfiguration(
        name: name,
        registration: registration.cast<TfLiteRegistration>(),
        minVersion: minVersion,
        maxVersion: maxVersion,
      ),
    );
    tfliteBinding.TfLiteInterpreterOptionsAddCustomOp(
      _options,
      opName,
      registration.cast<TfLiteRegistration>(),
      minVersion,
      maxVersion,
    );
  }

  /// Registers MediaPipe custom ops (like Convolution2DTransposeBias).
  ///
  /// Call this before creating an interpreter for MediaPipe models that use
  /// custom operations (e.g., Selfie Segmentation).
  ///
  /// Example:
  /// ```dart
  /// final options = InterpreterOptions();
  /// options.addMediaPipeCustomOps();
  /// final interpreter = await Interpreter.fromAsset('model.tflite', options: options);
  /// ```
  void addMediaPipeCustomOps() {
    checkState(!_deleted, message: 'InterpreterOptions already deleted.');
    TransposeConvBiasOp.registerWithOptions(_options);
    _hasMediaPipeCustomOps = true;
  }

  /// Creates equivalent options without hardware delegates.
  ///
  /// Used when interpreter creation cannot apply a configured delegate. CPU
  /// fallback must retain thread tuning and custom-op registrations rather
  /// than retrying with an empty options object.
  InterpreterOptions copyWithoutDelegates() {
    checkState(!_deleted, message: 'InterpreterOptions already deleted.');
    final copy = InterpreterOptions();
    final threads = _threads;
    if (threads != null) copy.threads = threads;
    if (_hasMediaPipeCustomOps) copy.addMediaPipeCustomOps();
    for (final customOp in _customOps) {
      copy.addCustomOp(
        name: customOp.name,
        registration: customOp.registration,
        minVersion: customOp.minVersion,
        maxVersion: customOp.maxVersion,
      );
    }
    return copy;
  }
}
