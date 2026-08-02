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
import '../ffi/helper.dart';
import 'tensor.dart';
import 'tflite_status.dart';

/// A runner for a specific model signature, enabling named tensor access.
///
/// Signatures allow models to expose multiple named entry points. This is the
/// foundation for on-device training, where a model typically exposes:
///
/// - `train`:       runs one training step (forward pass + backprop + optimizer)
/// - `infer`:       runs inference / prediction
/// - `get_weights`: reads current model weights into Dart buffers
/// - `set_weights`: writes weight values from Dart buffers into the model
///
/// ## On-device training example
///
/// ```dart
/// // Load a model that was exported with training signatures.
/// final interpreter = await Interpreter.fromAsset('trainable_model.tflite');
///
/// // Optionally inspect available signatures:
/// print(interpreter.signatureKeys); // ['train', 'infer', 'get_weights', 'set_weights']
///
/// // --- Training loop ---
/// final trainRunner = interpreter.getSignatureRunner('train');
///
/// for (int epoch = 0; epoch < 10; epoch++) {
///   final lossBuffer = Float32List(1);
///   trainRunner.run(
///     {'x': imageData, 'y': labels},
///     {'loss': lossBuffer},
///   );
///   print('Epoch $epoch loss: ${lossBuffer[0]}');
/// }
/// trainRunner.close();
///
/// // --- Inference ---
/// final inferRunner = interpreter.getSignatureRunner('infer');
/// final predictions = Float32List(10);
/// inferRunner.run({'x': testImage}, {'output': predictions});
/// inferRunner.close();
///
/// // --- Weight persistence (get/set) ---
/// final getRunner = interpreter.getSignatureRunner('get_weights');
/// final w = [[0.0]];
/// final b = [0.0];
/// getRunner.run({}, {'w': w, 'b': b});
/// getRunner.close();
///
/// // Restore weights into a fresh interpreter:
/// final setRunner = freshInterpreter.getSignatureRunner('set_weights');
/// setRunner.run({'w': w, 'b': b}, {});
/// setRunner.close();
///
/// interpreter.close();
/// ```
///
/// ## Building a training-capable model (Python)
///
/// Export a LiteRT model with `train`, `infer`, `get_weights`, and
/// `set_weights` signatures and convert with resource variables enabled:
///
/// ```python
/// converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
/// converter.experimental_enable_resource_variables = True
/// tflite_model = converter.convert()
/// ```
class SignatureRunner {
  final Pointer<TfLiteSignatureRunner> _runner;
  bool _closed = false;
  bool _allocated = false;
  int _lastInferenceDurationMicroseconds = 0;
  // Reused: monotonic and allocation-free, unlike DateTime.now() per run.
  final Stopwatch _inferenceStopwatch = Stopwatch();

  /// Creates a [SignatureRunner] from a raw native pointer.
  ///
  /// Typically obtained via [Interpreter.getSignatureRunner].
  SignatureRunner(this._runner) {
    checkArgument(
      isNotNull(_runner),
      message: 'SignatureRunner pointer is null.',
    );
  }

  // ---------------------------------------------------------------------------
  // Input tensors
  // ---------------------------------------------------------------------------

  /// The number of input tensors for this signature.
  int get inputCount =>
      tfliteBinding.TfLiteSignatureRunnerGetInputCount(_runner);

  /// Returns the name of the input tensor at [index].
  String getInputName(int index) {
    return tfliteBinding.TfLiteSignatureRunnerGetInputName(
      _runner,
      index,
    ).cast<Utf8>().toDartString();
  }

  /// The names of all input tensors for this signature.
  List<String> get inputNames =>
      List.generate(inputCount, getInputName, growable: false);

  // Tensor handles cached by name. Resolving a tensor costs a UTF-8
  // encode + FFI lookup per call, and run() resolves each input twice;
  // handles are stable between allocations, so cache until
  // resizeInputTensor/allocateTensors.
  final Map<String, Tensor> _inputTensorCache = {};
  final Map<String, Tensor> _outputTensorCache = {};

  void _invalidateTensorCaches() {
    _inputTensorCache.clear();
    _outputTensorCache.clear();
  }

  Tensor _getTensorByName(
    String name,
    Pointer<TfLiteTensor> Function(
      Pointer<TfLiteSignatureRunner>,
      Pointer<Char>,
    )
    nativeGetter,
    String kind,
    List<String> Function() validNames,
  ) {
    final namePtr = name.toNativeUtf8();
    try {
      final tensor = nativeGetter(_runner, namePtr.cast());
      if (!isNotNull(tensor)) {
        // Valid names are resolved only on failure: building the list costs
        // one FFI call plus a string decode per tensor.
        throw ArgumentError(
          '$kind tensor "$name" not found. '
          'Valid $kind names: ${validNames().join(', ')}',
        );
      }
      return Tensor(tensor);
    } finally {
      calloc.free(namePtr);
    }
  }

  /// Returns the input [Tensor] identified by [name].
  ///
  /// Throws [ArgumentError] if [name] is not a valid input name.
  Tensor getInputTensor(String name) =>
      _inputTensorCache[name] ??= _getTensorByName(
        name,
        tfliteBinding.TfLiteSignatureRunnerGetInputTensor,
        'Input',
        () => inputNames,
      );

  /// Returns all input tensors for this signature.
  List<Tensor> getInputTensors() =>
      inputNames.map(getInputTensor).toList(growable: false);

  /// Resizes the input tensor identified by [name] to [shape].
  ///
  /// [allocateTensors] must be called again after any resize before invoking.
  void resizeInputTensor(String name, List<int> shape) {
    final namePtr = name.toNativeUtf8();
    try {
      final dimensionSize = shape.length;
      final dimensions = calloc<Int>(dimensionSize);
      try {
        final externalTypedData = dimensions.cast<Int32>().asTypedList(
          dimensionSize,
        );
        externalTypedData.setRange(0, dimensionSize, shape);
        final status = tfliteBinding.TfLiteSignatureRunnerResizeInputTensor(
          _runner,
          namePtr.cast(),
          dimensions,
          dimensionSize,
        );
        checkTfLiteStatus('TfLiteSignatureRunnerResizeInputTensor', status);
        _allocated = false;
        _invalidateTensorCaches();
      } finally {
        calloc.free(dimensions);
      }
    } finally {
      calloc.free(namePtr);
    }
  }

  // ---------------------------------------------------------------------------
  // Allocation & invocation
  // ---------------------------------------------------------------------------

  /// Allocates memory for all tensors in this signature.
  ///
  /// Must be called after [Interpreter.getSignatureRunner] and after any
  /// [resizeInputTensor] calls, before the first [invoke].
  void allocateTensors() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    checkTfLiteStatus(
      'TfLiteSignatureRunnerAllocateTensors',
      tfliteBinding.TfLiteSignatureRunnerAllocateTensors(_runner),
    );
    _allocated = true;
    // Allocation can relocate tensor structs (e.g. first-time delegate
    // application grows the tensors arena), so cached handles may dangle.
    _invalidateTensorCaches();
  }

  /// Runs this signature.
  ///
  /// [allocateTensors] must have been called before invoking. Use the
  /// higher-level [run] method which handles allocation automatically.
  void invoke() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    checkState(
      _allocated,
      message: 'Tensors not allocated. Call allocateTensors() before invoke().',
    );
    checkTfLiteStatus(
      'TfLiteSignatureRunnerInvoke',
      tfliteBinding.TfLiteSignatureRunnerInvoke(_runner),
    );
  }

  /// Attempts to cancel the currently executing [invoke] call.
  ///
  /// Returns `true` if cancellation was successfully requested. Has no effect
  /// if no invocation is in progress. Cancellation is not guaranteed, the
  /// model may complete before the request is processed.
  bool cancel() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    return tfliteBinding.TfLiteSignatureRunnerCancel(_runner) ==
        TfLiteStatus.kTfLiteOk;
  }

  // ---------------------------------------------------------------------------
  // Output tensors
  // ---------------------------------------------------------------------------

  /// The number of output tensors for this signature.
  int get outputCount =>
      tfliteBinding.TfLiteSignatureRunnerGetOutputCount(_runner);

  /// Returns the name of the output tensor at [index].
  String getOutputName(int index) {
    return tfliteBinding.TfLiteSignatureRunnerGetOutputName(
      _runner,
      index,
    ).cast<Utf8>().toDartString();
  }

  /// The names of all output tensors for this signature.
  List<String> get outputNames =>
      List.generate(outputCount, getOutputName, growable: false);

  /// Returns the output [Tensor] identified by [name].
  ///
  /// Throws [ArgumentError] if [name] is not a valid output name.
  Tensor getOutputTensor(String name) =>
      _outputTensorCache[name] ??= _getTensorByName(
        name,
        tfliteBinding.TfLiteSignatureRunnerGetOutputTensor,
        'Output',
        () => outputNames,
      );

  /// Returns all output tensors for this signature.
  List<Tensor> getOutputTensors() =>
      outputNames.map(getOutputTensor).toList(growable: false);

  // ---------------------------------------------------------------------------
  // High-level run API
  // ---------------------------------------------------------------------------

  /// Runs this signature with named [inputs], writing results to [outputs].
  ///
  /// Automatically resizes input tensors if shapes differ, allocates tensors
  /// if needed, and copies output data into the provided objects.
  ///
  /// [inputs], map from input tensor name to data (List, Uint8List, etc.)
  /// [outputs], map from output tensor name to a pre-allocated object that
  ///             will receive the output data (List, Float32List, etc.).
  ///             Pass an empty map if the signature produces no outputs you
  ///             need to read (e.g. the `set_weights` signature).
  ///
  /// Example, training step:
  /// ```dart
  /// final lossBuffer = Float32List(1);
  /// trainRunner.run(
  ///   {'x': imageData, 'y': labels},
  ///   {'loss': lossBuffer},
  /// );
  /// ```
  ///
  /// Example, restore weights:
  /// ```dart
  /// setWeightsRunner.run({'w': w, 'b': b}, {});
  /// ```
  void run(Map<String, Object> inputs, Map<String, Object> outputs) {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    // Resize input tensors if shapes differ.
    for (final entry in inputs.entries) {
      final tensor = getInputTensor(entry.key);
      final newShape = tensor.getInputShapeIfDifferent(entry.value);
      if (newShape != null) {
        resizeInputTensor(entry.key, newShape);
      }
    }

    if (!_allocated) {
      allocateTensors();
    }

    // Copy input data into native tensors.
    for (final entry in inputs.entries) {
      getInputTensor(entry.key).setTo(entry.value);
    }

    _inferenceStopwatch
      ..reset()
      ..start();
    invoke();
    _inferenceStopwatch.stop();
    _lastInferenceDurationMicroseconds =
        _inferenceStopwatch.elapsedMicroseconds;

    // Copy output data back to Dart objects.
    for (final entry in outputs.entries) {
      getOutputTensor(entry.key).copyTo(entry.value);
    }
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /// Destroys this signature runner and releases native resources.
  ///
  /// Must be called before the [Interpreter] that created this runner is
  /// closed. After calling [close], this object must not be used.
  void close() {
    checkState(!_closed, message: 'SignatureRunner is already closed.');
    tfliteBinding.TfLiteSignatureRunnerDelete(_runner);
    _closed = true;
  }

  /// Whether [close] has been called on this runner.
  bool get isClosed => _closed;

  /// Duration of the most recent invocation in microseconds.
  ///
  /// Only meaningful after at least one call to [run]. Calling [invoke]
  /// directly does not update this value.
  int get lastInferenceDurationMicroseconds =>
      _lastInferenceDurationMicroseconds;

  /// Duration of the most recent native invocation in microseconds.
  ///
  /// Deprecated in favor of [lastInferenceDurationMicroseconds], which uses a
  /// platform-neutral name and consistent microseconds casing.
  @Deprecated(
    'Use lastInferenceDurationMicroseconds instead. '
    'This alias will be removed in a future major release.',
  )
  int get lastNativeInferenceDurationMicroSeconds =>
      lastInferenceDurationMicroseconds;

  @override
  String toString() => _closed
      ? 'SignatureRunner{closed: true}'
      : 'SignatureRunner{inputs: $inputNames, outputs: $outputNames, closed: false}';
}
