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

import 'dart:typed_data';

import 'compiled_model_types.dart';

export 'compiled_model_types.dart';

/// LiteRT Next CompiledModel inference API.
class CompiledModel {
  CompiledModel._();

  /// Creates a compiled model from a model file.
  static CompiledModel fromFile(
    String path, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Creates a compiled model from model bytes.
  static CompiledModel fromBuffer(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Creates a compiled model from model bytes without requiring synchronous
  /// compilation.
  static Future<CompiledModel> fromBufferAsync(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Creates a compiled model from [bytes], preferring GPU with a CPU fallback.
  static CompiledModel fromBufferWithGpuFallback(
    Uint8List bytes, {
    bool forceCpu = false,
    Precision precision = Precision.fp32,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
    void Function(Object error)? onFallback,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Creates a compiled model from [bytes], preferring GPU with a CPU
  /// fallback, without requiring synchronous compilation.
  static Future<CompiledModel> fromBufferWithGpuFallbackAsync(
    Uint8List bytes, {
    bool forceCpu = false,
    Precision precision = Precision.fp32,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
    void Function(Object error)? onFallback,
  }) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Tensor buffer allocation mode used by this model.
  TensorBufferMode get tensorBufferMode => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Accelerators this model was compiled with.
  Set<Accelerator> get accelerators => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Whether the whole graph runs on a selected hardware accelerator.
  bool get isFullyAccelerated => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Number of model input tensors.
  int get inputCount => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Number of model output tensors.
  int get outputCount => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Byte size of each input tensor buffer.
  List<int> get inputByteSizes => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Byte size of each output tensor buffer.
  List<int> get outputByteSizes => throw UnsupportedError(
    'CompiledModel is not supported on this platform.',
  );

  /// Runs inference with Float32 input tensors and returns Float32 outputs.
  List<Float32List> run(List<Float32List> inputs) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Runs inference asynchronously when the selected accelerator supports it.
  Future<List<Float32List>> runAsync(List<Float32List> inputs) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Writes input [index] directly into a host-memory tensor buffer.
  void writeInput(int index, void Function(Float32List input) write) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Runs inference using inputs previously written with [writeInput].
  void dispatch() {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Runs inference asynchronously using inputs previously written with
  /// [writeInput].
  Future<void> dispatchAsync() {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Reads output [index] directly from a host-memory tensor buffer.
  R readOutput<R>(int index, R Function(Float32List output) read) {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }

  /// Releases native CompiledModel resources.
  void close() {
    throw UnsupportedError('CompiledModel is not supported on this platform.');
  }
}
