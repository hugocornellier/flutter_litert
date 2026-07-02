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
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import '../bindings/bindings.dart';
import '../bindings/tensorflow_lite_bindings_generated.dart';

import '../ffi/helper.dart';
import '../util/flutter_asset_utils_stub.dart'
    if (dart.library.ui) '../util/flutter_asset_utils.dart';
import 'interpreter_options.dart';
import 'model.dart';
import 'signature_runner.dart';
import 'tensor.dart';

/// LiteRT interpreter for running inference on a model.
class Interpreter {
  final Pointer<TfLiteInterpreter> _interpreter;
  final bool _hasActiveDelegate;
  Pointer<Uint8>? _modelBuffer;
  bool _deleted = false;
  bool _allocated = false;
  int _lastInferenceDurationMicroseconds = 0;
  // Reused: monotonic and allocation-free, unlike DateTime.now() per run.
  final Stopwatch _inferenceStopwatch = Stopwatch();

  /// Returns the LiteRT runtime version string.
  static String get version =>
      tfliteBinding.TfLiteVersion().cast<Utf8>().toDartString();

  List<Tensor>? _inputTensors;
  List<Tensor>? _outputTensors;

  int? _inputTensorsCount;
  int? _outputTensorsCount;

  /// Duration of the last inference call in microseconds.
  int get lastInferenceDurationMicroseconds =>
      _lastInferenceDurationMicroseconds;

  /// Whether interpreter creation successfully applied a hardware delegate.
  ///
  /// This can be false even when the supplied [InterpreterOptions] contained a
  /// delegate, because delegate application is best-effort and creation may
  /// retry with equivalent CPU-only options.
  bool get hasActiveDelegate => _hasActiveDelegate;

  /// Duration of the last native inference call in microseconds.
  ///
  /// Deprecated in favor of [lastInferenceDurationMicroseconds], which uses a
  /// platform-neutral name and consistent microseconds casing.
  @Deprecated(
    'Use lastInferenceDurationMicroseconds instead. '
    'This alias will be removed in a future major release.',
  )
  int get lastNativeInferenceDurationMicroSeconds =>
      lastInferenceDurationMicroseconds;

  Interpreter._(
    this._interpreter, {
    required bool hasActiveDelegate,
    bool skipAllocate = false,
  }) : _hasActiveDelegate = hasActiveDelegate {
    if (!skipAllocate) {
      allocateTensors();
    }
  }

  /// Creates interpreter from model
  ///
  /// Throws [ArgumentError] is unsuccessful.
  factory Interpreter._create(Model model, {InterpreterOptions? options}) {
    var interpreter = tfliteBinding.TfLiteInterpreterCreate(
      model.base,
      options?.base ?? cast<TfLiteInterpreterOptions>(nullptr),
    );
    var hasActiveDelegate =
        isNotNull(interpreter) && (options?.hasDelegate ?? false);
    if (!isNotNull(interpreter) && (options?.hasDelegate ?? false)) {
      // The configured delegate could not be applied to this model/runtime.
      // Preserve all non-delegate options when retrying on CPU.
      stderr.writeln(
        'flutter_litert: interpreter creation failed with the configured '
        'delegate; falling back to CPU.',
      );
      final fallbackOptions = options!.copyWithoutDelegates();
      try {
        interpreter = tfliteBinding.TfLiteInterpreterCreate(
          model.base,
          fallbackOptions.base,
        );
      } finally {
        fallbackOptions.delete();
      }
      hasActiveDelegate = false;
    }
    checkArgument(
      isNotNull(interpreter),
      message: 'Unable to create interpreter.',
    );
    // Transfer buffer ownership: the interpreter references model weight data
    // directly via zero-copy, so the buffer must stay alive until the
    // interpreter is destroyed.
    return Interpreter._(interpreter, hasActiveDelegate: hasActiveDelegate)
      .._modelBuffer = model.detachBuffer();
  }

  /// Creates [Interpreter] from a model file
  ///
  /// Throws [ArgumentError] if unsuccessful.
  ///
  /// Example:
  ///
  /// ```dart
  /// final dataFile = await getFile('assets/your_model.tflite');
  /// final interpreter = Interpreter.fromFile(dataFile);
  ///
  /// Future<File> getFile(String fileName) async {
  ///   final appDir = await getTemporaryDirectory();
  ///   final appPath = appDir.path;
  ///   final fileOnDevice = File('$appPath/$fileName');
  ///   final rawAssetFile = await rootBundle.load(fileName);
  ///   final rawBytes = rawAssetFile.buffer.asUint8List();
  ///   await fileOnDevice.writeAsBytes(rawBytes, flush: true);
  ///   return fileOnDevice;
  /// }
  /// ```
  factory Interpreter.fromFile(File modelFile, {InterpreterOptions? options}) {
    final model = Model.fromFile(modelFile.path);
    try {
      return Interpreter._create(model, options: options);
    } finally {
      model.delete();
    }
  }

  /// Creates interpreter from a [buffer].
  ///
  /// Prefer [fromBytes] for new cross-platform code. This synchronous
  /// constructor remains for `tflite_flutter` compatibility on native
  /// platforms.
  ///
  /// Throws [ArgumentError] if unsuccessful.
  ///
  /// Example:
  ///
  /// ```dart
  ///   final buffer = await getBuffer('assets/your_model.tflite');
  ///   final interpreter = Interpreter.fromBuffer(buffer);
  ///
  ///   Future<Uint8List> getBuffer(String filePath) async {
  ///       final rawAssetFile = await rootBundle.load(filePath);
  ///       final rawBytes = rawAssetFile.buffer.asUint8List();
  ///       return rawBytes;
  ///   }
  /// ```
  factory Interpreter.fromBuffer(
    Uint8List buffer, {
    InterpreterOptions? options,
  }) {
    final model = Model.fromBuffer(buffer);
    try {
      return Interpreter._create(model, options: options);
    } finally {
      model.delete();
    }
  }

  /// Creates an interpreter from raw model [bytes].
  ///
  /// This is the async, cross-platform spelling of [fromBuffer]. On native
  /// platforms it completes immediately after constructing the interpreter.
  static Future<Interpreter> fromBytes(
    Uint8List bytes, {
    InterpreterOptions? options,
  }) async {
    return Interpreter.fromBuffer(bytes, options: options);
  }

  /// Creates interpreter from a [assetName]
  ///
  /// Place your `.tflite` file in your assets folder.
  ///
  /// Example:
  ///
  /// ```dart
  /// final interpreter = await tfl.Interpreter.fromAsset('assets/your_model.tflite');
  /// ```
  static Future<Interpreter> fromAsset(
    String assetName, {
    InterpreterOptions? options,
  }) async {
    Uint8List buffer = await loadAssetBytes(assetName);
    return Interpreter.fromBuffer(buffer, options: options);
  }

  /// Creates interpreter from an address.
  ///
  /// Typically used for passing interpreter between isolates.
  /// [allocated] defaults to true because fromAddress is typically called
  /// after tensors have already been allocated on the original interpreter.
  /// This avoids redundant (and potentially thread-unsafe) allocateTensors()
  /// calls when used across isolate boundaries.
  factory Interpreter.fromAddress(
    int address, {
    bool allocated = true,
    bool deleted = false,
    bool hasActiveDelegate = false,
  }) {
    final interpreter = Pointer<TfLiteInterpreter>.fromAddress(address);
    return Interpreter._(
        interpreter,
        hasActiveDelegate: hasActiveDelegate,
        skipAllocate: allocated,
      )
      .._deleted = deleted
      .._allocated = allocated;
  }

  /// Destroys the interpreter instance.
  void close() {
    checkState(!_deleted, message: 'Interpreter already deleted.');
    tfliteBinding.TfLiteInterpreterDelete(_interpreter);
    if (_modelBuffer != null) {
      calloc.free(_modelBuffer!);
      _modelBuffer = null;
    }
    _deleted = true;
  }

  /// Updates allocations for all tensors.
  void allocateTensors() {
    checkState(
      tfliteBinding.TfLiteInterpreterAllocateTensors(_interpreter) ==
          TfLiteStatus.kTfLiteOk,
    );
    _allocated = true;
  }

  /// Runs inference for the loaded graph.
  void invoke() {
    checkState(_allocated, message: 'Interpreter not allocated.');
    checkState(
      tfliteBinding.TfLiteInterpreterInvoke(_interpreter) ==
          TfLiteStatus.kTfLiteOk,
    );
  }

  /// Run for single input and output
  void run(Object input, Object output) {
    final outputCount = _outputTensorsCount ??=
        tfliteBinding.TfLiteInterpreterGetOutputTensorCount(_interpreter);
    if (outputCount != 1) {
      // Preserve the map path and its missing-output failure mode.
      runForMultipleInputs([input], <int, Object>{0: output});
      return;
    }
    // Single-output fast path: same steps as runForMultipleInputs without
    // allocating and indexing the one-entry output map.
    _inputTensors = null;
    _outputTensors = null;
    runInference([input]);
    Tensor(
      tfliteBinding.TfLiteInterpreterGetOutputTensor(_interpreter, 0),
    ).copyTo(output);
  }

  /// Run for multiple inputs and outputs
  void runForMultipleInputs(List<Object> inputs, Map<int, Object> outputs) {
    if (outputs.isEmpty) {
      throw ArgumentError('Input error: Outputs should not be null or empty.');
    }
    // Invalidate cached tensor handles before each run so we always use
    // fresh pointers from the interpreter. TFLite may relocate internal
    // tensor storage between invocations (e.g. XNNPACK workspace reuse).
    _inputTensors = null;
    _outputTensors = null;
    runInference(inputs);
    // Fresh per-index pointers without rebuilding the wrapper list (the
    // count is stable between allocateTensors calls, the pointers are not).
    final outputCount = _outputTensorsCount ??=
        tfliteBinding.TfLiteInterpreterGetOutputTensorCount(_interpreter);
    for (var i = 0; i < outputCount; i++) {
      Tensor(
        tfliteBinding.TfLiteInterpreterGetOutputTensor(_interpreter, i),
      ).copyTo(outputs[i]!);
    }
  }

  /// Just run inference
  void runInference(List<Object> inputs) {
    if (inputs.isEmpty) {
      throw ArgumentError('Input error: Inputs should not be null or empty.');
    }

    final inputCount = _inputTensorsCount ??=
        tfliteBinding.TfLiteInterpreterGetInputTensorCount(_interpreter);
    if (inputs.length > inputCount) {
      throw RangeError.range(inputs.length - 1, 0, inputCount - 1, 'inputs');
    }

    // Steady-state fast path: allocated and no resize needed. One pass with
    // a fresh pointer per index, no wrapper list churn. Any resize or
    // missing allocation falls through to the two-pass path below, which
    // re-reads every pointer because resize/allocate relocates tensors.
    var deferred = !_allocated;
    for (int i = 0; i < inputs.length && !deferred; i++) {
      final tensor = Tensor(
        tfliteBinding.TfLiteInterpreterGetInputTensor(_interpreter, i),
      );
      final newShape = tensor.getInputShapeIfDifferent(inputs[i]);
      if (newShape != null) {
        resizeInputTensor(i, newShape);
        deferred = true;
      } else {
        tensor.setTo(inputs[i]);
      }
    }

    if (deferred) {
      for (int i = 0; i < inputs.length; i++) {
        final tensor = Tensor(
          tfliteBinding.TfLiteInterpreterGetInputTensor(_interpreter, i),
        );
        final newShape = tensor.getInputShapeIfDifferent(inputs[i]);
        if (newShape != null) {
          resizeInputTensor(i, newShape);
        }
      }

      if (!_allocated) {
        allocateTensors();
      }

      for (int i = 0; i < inputs.length; i++) {
        Tensor(
          tfliteBinding.TfLiteInterpreterGetInputTensor(_interpreter, i),
        ).setTo(inputs[i]);
      }
    }

    _inferenceStopwatch
      ..reset()
      ..start();
    invoke();
    _inferenceStopwatch.stop();
    _lastInferenceDurationMicroseconds =
        _inferenceStopwatch.elapsedMicroseconds;
  }

  List<Tensor> _buildTensorList(int count, Tensor Function(int) getter) =>
      List.generate(count, getter, growable: false);

  /// Gets all input tensors associated with the model.
  List<Tensor> getInputTensors() {
    return _inputTensors ??= _buildTensorList(
      tfliteBinding.TfLiteInterpreterGetInputTensorCount(_interpreter),
      (i) => Tensor(
        tfliteBinding.TfLiteInterpreterGetInputTensor(_interpreter, i),
      ),
    );
  }

  /// Gets all output tensors associated with the model.
  List<Tensor> getOutputTensors() {
    return _outputTensors ??= _buildTensorList(
      tfliteBinding.TfLiteInterpreterGetOutputTensorCount(_interpreter),
      (i) => Tensor(
        tfliteBinding.TfLiteInterpreterGetOutputTensor(_interpreter, i),
      ),
    );
  }

  /// Resize input tensor for the given tensor index. `allocateTensors` must be called again afterward.
  void resizeInputTensor(int tensorIndex, List<int> shape) {
    final dimensionSize = shape.length;
    final dimensions = calloc<Int>(dimensionSize);
    final externalTypedData = dimensions.cast<Int32>().asTypedList(
      dimensionSize,
    );
    externalTypedData.setRange(0, dimensionSize, shape);
    final status = tfliteBinding.TfLiteInterpreterResizeInputTensor(
      _interpreter,
      tensorIndex,
      dimensions,
      dimensionSize,
    );
    calloc.free(dimensions);
    checkState(status == TfLiteStatus.kTfLiteOk);
    _inputTensors = null;
    _outputTensors = null;
    _inputTensorsCount = null;
    _outputTensorsCount = null;
    _allocated = false;
  }

  /// Gets the input Tensor for the provided input index.
  Tensor getInputTensor(int index) {
    _inputTensorsCount ??= tfliteBinding.TfLiteInterpreterGetInputTensorCount(
      _interpreter,
    );
    if (index < 0 || index >= _inputTensorsCount!) {
      throw ArgumentError('Invalid input Tensor index: $index');
    }
    if (_inputTensors != null) {
      return _inputTensors![index];
    }

    final inputTensor = Tensor(
      tfliteBinding.TfLiteInterpreterGetInputTensor(_interpreter, index),
    );
    return inputTensor;
  }

  /// Gets the output Tensor for the provided output index.
  Tensor getOutputTensor(int index) {
    _outputTensorsCount ??= tfliteBinding.TfLiteInterpreterGetOutputTensorCount(
      _interpreter,
    );
    if (index < 0 || index >= _outputTensorsCount!) {
      throw ArgumentError('Invalid output Tensor index: $index');
    }
    if (_outputTensors != null) {
      return _outputTensors![index];
    }
    final outputTensor = Tensor(
      tfliteBinding.TfLiteInterpreterGetOutputTensor(_interpreter, index),
    );
    return outputTensor;
  }

  int _findTensorIndex(String opName, List<Tensor> tensors, String kind) {
    for (var i = 0; i < tensors.length; i++) {
      if (tensors[i].name == opName) return i;
    }
    throw ArgumentError(
      "$kind error: '$opName' is not a valid name for any $kind.",
    );
  }

  /// Gets index of an input given the op name of the input.
  int getInputIndex(String opName) =>
      _findTensorIndex(opName, getInputTensors(), 'Input');

  /// Gets index of an output given the op name of the output.
  int getOutputIndex(String opName) =>
      _findTensorIndex(opName, getOutputTensors(), 'Output');

  /// Resets all variable tensors to their default values.
  void resetVariableTensors() {
    checkState(
      !_deleted,
      message: 'Should not access interpreter after it has been closed.',
    );
    tfliteBinding.TfLiteInterpreterResetVariableTensors(_interpreter);
  }

  /// Returns the number of variable (trainable) tensors in the model.
  int getVariableTensorCount() {
    checkState(
      !_deleted,
      message: 'Should not access interpreter after it has been closed.',
    );
    return tfliteBinding.TfLiteInterpreterGetVariableTensorCount(_interpreter);
  }

  /// Gets the variable (trainable) tensor at the given [index].
  Tensor getVariableTensor(int index) {
    checkState(
      !_deleted,
      message: 'Should not access interpreter after it has been closed.',
    );
    final count = getVariableTensorCount();
    if (index < 0 || index >= count) {
      throw ArgumentError('Invalid variable Tensor index: $index');
    }
    return Tensor(
      tfliteBinding.TfLiteInterpreterGetVariableTensor(_interpreter, index),
    );
  }

  /// Returns the address to the interpreter
  int get address => _interpreter.address;

  /// Whether tensors have been allocated.
  bool get isAllocated => _allocated;

  /// Whether this interpreter has been closed.
  bool get isDeleted => _deleted;

  // ---------------------------------------------------------------------------
  // Signature / SignatureRunner APIs
  // ---------------------------------------------------------------------------

  /// Returns the number of signatures defined in the model.
  ///
  /// Training-capable models typically expose multiple signatures:
  /// `train`, `infer`, `get_weights`, and `set_weights`.
  int get signatureCount =>
      tfliteBinding.TfLiteInterpreterGetSignatureCount(_interpreter);

  /// Returns the key (name) of the signature at [index].
  ///
  /// Use [signatureKeys] to list all signature keys at once.
  String getSignatureKey(int index) {
    return tfliteBinding.TfLiteInterpreterGetSignatureKey(
      _interpreter,
      index,
    ).cast<Utf8>().toDartString();
  }

  /// Returns the keys of all signatures defined in the model.
  ///
  /// For a training-capable model this typically returns:
  /// `['train', 'infer', 'get_weights', 'set_weights']`.
  List<String> get signatureKeys =>
      List.generate(signatureCount, getSignatureKey, growable: false);

  /// Returns a [SignatureRunner] for the signature identified by [signatureKey].
  ///
  /// The caller is responsible for calling [SignatureRunner.close] on the
  /// returned runner before this [Interpreter] is closed.
  ///
  /// Throws [ArgumentError] if [signatureKey] is not found in the model.
  ///
  /// Example, run a training step:
  /// ```dart
  /// final trainRunner = interpreter.getSignatureRunner('train');
  /// final lossBuffer = Float32List(1);
  /// trainRunner.run({'x': imageData, 'y': labels}, {'loss': lossBuffer});
  /// print('Loss: ${lossBuffer[0]}');
  /// trainRunner.close();
  /// ```
  SignatureRunner getSignatureRunner(String signatureKey) {
    final keyPtr = signatureKey.toNativeUtf8();
    try {
      final runner = tfliteBinding.TfLiteInterpreterGetSignatureRunner(
        _interpreter,
        keyPtr.cast(),
      );
      checkArgument(
        isNotNull(runner),
        message:
            'Signature "$signatureKey" not found. '
            'Available signatures: ${signatureKeys.join(', ')}',
      );
      return SignatureRunner(runner);
    } finally {
      calloc.free(keyPtr);
    }
  }
}
