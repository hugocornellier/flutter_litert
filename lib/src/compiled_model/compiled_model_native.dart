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

import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../bindings/litert_ffi.dart';
import '../bindings/litert_loader.dart';
import '../util/async_lock.dart';
import 'compiled_model_types.dart';
import 'litert_status.dart';

export 'compiled_model_types.dart';

const int _kLiteRtStatusOk = 0;
const int _kLiteRtHwAcceleratorCpu = 1;
const int _kLiteRtHwAcceleratorGpu = 2;
const int _kLiteRtHwAcceleratorNpu = 4;
const int _kLiteRtElementTypeFloat32 = 1;
const int _kLiteRtTensorBufferLockModeRead = 0;
const int _kLiteRtTensorBufferLockModeWrite = 1;
const int _kLiteRtDelegatePrecisionFp32 = 2;
const int _kLiteRtAnySize = 16;
const int _kLiteRtAnyValueOffset = 8;
const int _kLiteRtEnvOptionSize = 24;
const int _kLiteRtEnvOptionValueOffset = 8;
const int _kHostMemoryAlignment = 64;

/// LiteRT Next CompiledModel inference API.
class CompiledModel {
  CompiledModel._(
    this._rt,
    this._environment,
    this._options,
    this._model,
    this._modelBuffer,
    this._compiledModel,
    this._inputBuffers,
    this._outputBuffers,
    this._inputHostMemory,
    this._outputHostMemory,
    this._hostMemoryAllocations,
    this._inputByteSizes,
    this._outputByteSizes,
    this._tensorBufferMode,
    this._accelerators,
    this._gpuOptionsIdentifier,
  ) : _inputCount = _inputByteSizes.length,
      _outputCount = _outputByteSizes.length;

  final LiteRtBindings _rt;
  final Pointer<Void> _environment;
  final Pointer<Void> _options;
  final Pointer<Void> _model;
  final Pointer<Uint8>? _modelBuffer;
  final Pointer<Void> _compiledModel;
  final Pointer<Pointer<Void>> _inputBuffers;
  final Pointer<Pointer<Void>> _outputBuffers;
  final List<_HostMemoryAllocation?> _inputHostMemory;
  final List<_HostMemoryAllocation?> _outputHostMemory;
  final List<_HostMemoryAllocation> _hostMemoryAllocations;
  final List<int> _inputByteSizes;
  final List<int> _outputByteSizes;
  final TensorBufferMode _tensorBufferMode;
  final Set<Accelerator> _accelerators;
  final Pointer<Utf8>? _gpuOptionsIdentifier;
  final int _inputCount;
  final int _outputCount;

  bool _closed = false;

  // Per-dispatch native out-param, allocated once instead of per call.
  late final Pointer<Pointer<Void>> _lockScratch = calloc<Pointer<Void>>();

  // Async dispatch state: whole runAsync cycles (write, dispatch, read) are
  // serialized because they share this model's native I/O buffers, and the
  // blocking native call runs on a lazily-spawned helper isolate.
  final AsyncLock _runAsyncLock = AsyncLock();
  _AsyncDispatcher? _asyncDispatcher;
  bool _dispatchInFlight = false;

  /// Number of input tensors for the default signature.
  int get inputCount => _inputCount;

  /// Number of output tensors for the default signature.
  int get outputCount => _outputCount;

  /// Byte size of each input tensor's managed buffer, index-aligned with [run].
  List<int> get inputByteSizes => List.unmodifiable(_inputByteSizes);

  /// Byte size of each output tensor's managed buffer, index-aligned with [run]'s result.
  List<int> get outputByteSizes => List.unmodifiable(_outputByteSizes);

  /// Tensor buffer allocation mode used by this model.
  TensorBufferMode get tensorBufferMode => _tensorBufferMode;

  /// Accelerators this model was compiled with.
  ///
  /// This is the set requested at creation that compilation succeeded with.
  /// Compiling successfully is not the same as running on that hardware, so
  /// use [isFullyAccelerated] to find out whether the accelerator actually
  /// took the graph.
  Set<Accelerator> get accelerators => Set.unmodifiable(_accelerators);

  /// Whether the **whole** graph runs on a selected hardware accelerator.
  ///
  /// True is a strong signal: every op was accepted, so nothing silently fell
  /// back. Per LiteRT's contract it is true when any one selected accelerator
  /// takes the entire model, so requesting `{gpu, npu}` and landing wholly on
  /// the GPU still reports true.
  ///
  /// **False is ambiguous, and deliberately not a fallback detector.** It means
  /// "not everything was accelerated", which covers both partial delegation and
  /// none at all. Measured on macOS arm64, every model in the cat/dog/animal
  /// pipelines reports false under `{gpu, cpu}`, including ones where the GPU
  /// demonstrably did run: these graphs get partially delegated (a handful of
  /// ops to the GPU, the rest to the CPU), which is enough to make this false
  /// while the GPU is genuinely active.
  ///
  /// So do not use this to decide whether acceleration happened. To detect a
  /// silent CPU fallback, compare output against a bare-CPU reference and look
  /// for a *nonzero* deviation, which is what [verifyCompiledModel] does:
  /// bit-identical output means the accelerator contributed nothing.
  ///
  /// Reports false rather than throwing when the runtime cannot answer, so a
  /// diagnostic call never takes down a working model.
  bool get isFullyAccelerated {
    _ensureOpen();
    final out = calloc<Uint8>();
    try {
      final status = _rt.isFullyAccelerated(_compiledModel, out);
      if (status != _kLiteRtStatusOk) return false;
      return out.value != 0;
    } finally {
      calloc.free(out);
    }
  }

  /// Creates a compiled model from a model file.
  static CompiledModel fromFile(
    String path, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    return _fromSource(
      accelerators: accelerators,
      precision: precision,
      tensorBufferMode: tensorBufferMode,
      createModel: (rt) => _createModelFromFile(rt, path),
    );
  }

  /// Creates a compiled model from model bytes.
  ///
  /// The bytes are copied into a native buffer owned by this [CompiledModel] so
  /// the model source remains alive until [close] releases the LiteRT model.
  static CompiledModel fromBuffer(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) {
    return _fromSource(
      accelerators: accelerators,
      precision: precision,
      tensorBufferMode: tensorBufferMode,
      createModel: (rt) => _createModelFromBuffer(rt, bytes),
    );
  }

  /// Creates a compiled model from model bytes without requiring synchronous
  /// compilation.
  ///
  /// Portable alternative to [fromBuffer]: on the web, LiteRT.js compilation
  /// is Promise-based and only this variant is available, so code that must
  /// also run on the web should prefer it. On native platforms compilation
  /// still runs synchronously inside this call.
  static Future<CompiledModel> fromBufferAsync(
    Uint8List bytes, {
    Set<Accelerator> accelerators = const {Accelerator.cpu},
    Precision precision = Precision.fp16,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  }) async {
    return fromBuffer(
      bytes,
      accelerators: accelerators,
      precision: precision,
      tensorBufferMode: tensorBufferMode,
    );
  }

  /// Creates a compiled model from [bytes], preferring GPU with a CPU fallback.
  ///
  /// Requests `{gpu, cpu}` at [precision] (default [Precision.fp32], since
  /// pixel-space landmark/box coordinates lose accuracy in fp16 and weak desktop
  /// GPU drivers have produced wrong fp16 results); if GPU compilation throws
  /// (unsupported op, no GPU, driver bug) it retries CPU-only so callers always
  /// get a working model. Pass [forceCpu] to skip the GPU attempt entirely (e.g.
  /// for a detector whose detection counts are sensitive to GPU floating-point
  /// variance).
  ///
  /// [precision] applies to every path, including the [forceCpu] shortcut and
  /// the CPU retry, so a model never silently drops to a precision the caller
  /// did not ask for.
  ///
  /// [onFallback] is invoked with the GPU error when the CPU retry happens, so
  /// callers can log it; the library itself stays free of a logging dependency.
  /// Use [accelerators] (via [fromBuffer]) directly if you need a custom set.
  static CompiledModel fromBufferWithGpuFallback(
    Uint8List bytes, {
    bool forceCpu = false,
    Precision precision = Precision.fp32,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
    void Function(Object error)? onFallback,
  }) {
    if (forceCpu) {
      return fromBuffer(
        bytes,
        accelerators: const {Accelerator.cpu},
        precision: precision,
        tensorBufferMode: tensorBufferMode,
      );
    }
    try {
      return fromBuffer(
        bytes,
        accelerators: const {Accelerator.gpu, Accelerator.cpu},
        precision: precision,
        tensorBufferMode: tensorBufferMode,
      );
    } catch (e) {
      onFallback?.call(e);
      return fromBuffer(
        bytes,
        accelerators: const {Accelerator.cpu},
        precision: precision,
        tensorBufferMode: tensorBufferMode,
      );
    }
  }

  /// Creates a compiled model from [bytes], preferring GPU with a CPU
  /// fallback, without requiring synchronous compilation.
  ///
  /// Portable alternative to [fromBufferWithGpuFallback]: on the web only
  /// this variant is available. On native platforms compilation still runs
  /// synchronously inside this call.
  static Future<CompiledModel> fromBufferWithGpuFallbackAsync(
    Uint8List bytes, {
    bool forceCpu = false,
    Precision precision = Precision.fp32,
    TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
    void Function(Object error)? onFallback,
  }) async {
    return fromBufferWithGpuFallback(
      bytes,
      forceCpu: forceCpu,
      precision: precision,
      tensorBufferMode: tensorBufferMode,
      onFallback: onFallback,
    );
  }

  /// Process-wide (per-isolate) shared LiteRT environment.
  ///
  /// Creating a LiteRT environment spins up the full GPU/WebGPU stack (adapter
  /// enumeration, device + context creation, kernel cache), a cost of hundreds
  /// of milliseconds. Previously every [CompiledModel] created its own, so an
  /// app loading N models paid that cost N times. LiteRT environments are
  /// designed to be shared across many compiled models, so we create one lazily
  /// per isolate and reuse it. It is intentionally never destroyed: it lives for
  /// the lifetime of the isolate (a long-lived GPU context singleton).
  static Pointer<Void>? _sharedEnvironment;

  static Pointer<Void> _sharedEnvironmentOf(LiteRtBindings rt) =>
      _sharedEnvironment ??= _createEnvironment(rt);

  static CompiledModel _fromSource({
    required Set<Accelerator> accelerators,
    required Precision precision,
    required TensorBufferMode tensorBufferMode,
    required _ModelSource Function(LiteRtBindings rt) createModel,
  }) {
    _checkStructLayouts();
    final acceleratorMask = _acceleratorMask(accelerators);

    final rt = litert;
    Pointer<Void> environment = nullptr;
    Pointer<Void> options = nullptr;
    Pointer<Void> model = nullptr;
    Pointer<Uint8>? modelBuffer;
    Pointer<Void> compiledModel = nullptr;
    Pointer<Pointer<Void>>? inputBuffers;
    Pointer<Pointer<Void>>? outputBuffers;
    Pointer<Utf8>? gpuOptionsIdentifier;
    var createdInputBuffers = 0;
    var createdOutputBuffers = 0;
    final inputHostMemory = <_HostMemoryAllocation?>[];
    final outputHostMemory = <_HostMemoryAllocation?>[];
    final hostMemoryAllocations = <_HostMemoryAllocation>[];

    try {
      environment = _sharedEnvironmentOf(rt);
      options = _createOptions(rt, acceleratorMask);
      if (accelerators.contains(Accelerator.gpu) &&
          precision == Precision.fp32) {
        gpuOptionsIdentifier = _addGpuFp32Options(rt, options);
      }
      final source = createModel(rt);
      model = source.model;
      modelBuffer = source.modelBuffer;
      compiledModel = _createCompiledModel(rt, environment, model, options);

      final signature = _getModelSignature(rt, model);
      final inputCount = _getSignatureInputCount(rt, signature);
      final outputCount = _getSignatureOutputCount(rt, signature);

      inputBuffers = calloc<Pointer<Void>>(inputCount);
      final inputByteSizes = <int>[];
      for (var i = 0; i < inputCount; i++) {
        final buffer = _createInputBuffer(
          rt,
          environment,
          compiledModel,
          signature,
          i,
          inputByteSizes,
          tensorBufferMode,
          inputHostMemory,
          hostMemoryAllocations,
        );
        inputBuffers[i] = buffer;
        createdInputBuffers++;
      }

      outputBuffers = calloc<Pointer<Void>>(outputCount);
      final outputByteSizes = <int>[];
      final outputLayouts = calloc<Uint8>(
        outputCount * kLiteRtLayoutByteSize,
      ).cast<LiteRtLayout>();
      try {
        _check(
          'LiteRtGetCompiledModelOutputTensorLayouts',
          rt.getCompiledModelOutputTensorLayouts(
            compiledModel,
            0,
            outputCount,
            outputLayouts,
            1,
          ),
        );
        for (var i = 0; i < outputCount; i++) {
          final buffer = _createOutputBuffer(
            rt,
            environment,
            compiledModel,
            signature,
            i,
            (outputLayouts.cast<Uint8>() + i * kLiteRtLayoutByteSize)
                .cast<LiteRtLayout>(),
            outputByteSizes,
            tensorBufferMode,
            outputHostMemory,
            hostMemoryAllocations,
          );
          outputBuffers[i] = buffer;
          createdOutputBuffers++;
        }
      } finally {
        calloc.free(outputLayouts);
      }

      return CompiledModel._(
        rt,
        environment,
        options,
        model,
        modelBuffer,
        compiledModel,
        inputBuffers,
        outputBuffers,
        inputHostMemory,
        outputHostMemory,
        hostMemoryAllocations,
        inputByteSizes,
        outputByteSizes,
        tensorBufferMode,
        Set.of(accelerators),
        gpuOptionsIdentifier,
      );
    } catch (_) {
      _releaseNative(
        rt,
        inputBuffers: inputBuffers,
        inputCount: createdInputBuffers,
        outputBuffers: outputBuffers,
        outputCount: createdOutputBuffers,
        compiledModel: compiledModel,
        model: model,
        modelBuffer: modelBuffer,
        options: options,
        environment: environment,
        gpuOptionsIdentifier: gpuOptionsIdentifier,
        hostMemoryAllocations: hostMemoryAllocations,
      );
      rethrow;
    }
  }

  /// Runs inference with Float32 input tensors and returns Float32 outputs.
  ///
  /// Throws [StateError] while an async dispatch is in flight: the helper
  /// isolate is running the model against the same native I/O buffers.
  List<Float32List> run(List<Float32List> inputs) {
    _ensureOpen();
    _ensureNoAsyncDispatch('run');
    if (inputs.length != _inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected $_inputCount input tensors.',
      );
    }

    for (var i = 0; i < _inputCount; i++) {
      _writeInput(i, inputs[i]);
    }

    _dispatch();

    return List<Float32List>.generate(_outputCount, _readOutput);
  }

  void _dispatch() {
    _ensureOpen();
    _check(
      'LiteRtRunCompiledModel',
      _rt.runCompiledModel(
        _compiledModel,
        0,
        _inputCount,
        _inputBuffers,
        _outputCount,
        _outputBuffers,
      ),
    );
  }

  /// Runs inference without blocking the calling isolate.
  ///
  /// The blocking native call executes on a lazily-spawned, per-model helper
  /// isolate, so the calling isolate's event loop keeps servicing timers,
  /// microtasks, and UI work while the model runs. Concurrent calls are
  /// serialized in FIFO order because they share this model's native I/O
  /// buffers. Each call pays an isolate message round trip on top of the
  /// inference itself; prefer [run] when blocking is acceptable and the
  /// model is very fast, or when you are already calling from a background
  /// isolate you own (there the helper isolate only adds a hop).
  ///
  /// [inputs] are copied into the model's buffers when this call's turn in
  /// the queue arrives, not at call time; do not mutate them until the
  /// returned future completes.
  ///
  /// The helper isolate runs the model on a different thread than the one
  /// that compiled it. CPU and Apple Metal accelerators are safe; runAsync
  /// with thread-affine mobile GPU stacks (some Android GL/CL drivers) is
  /// unvalidated, prefer [run] there until it is.
  Future<List<Float32List>> runAsync(List<Float32List> inputs) async {
    _ensureOpen();
    if (inputs.length != _inputCount) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected $_inputCount input tensors.',
      );
    }

    return _runAsyncLock.run(() async {
      _ensureOpen();
      // A bare dispatchAsync does not go through this lock; refuse to write
      // over the buffers it is still running against.
      _ensureNoAsyncDispatch('runAsync');
      for (var i = 0; i < _inputCount; i++) {
        _writeInput(i, inputs[i]);
      }

      await _dispatchAsync();

      return List<Float32List>.generate(_outputCount, _readOutput);
    });
  }

  /// Writes input [index] directly into this model's host-memory tensor buffer.
  ///
  /// This is available only when [tensorBufferMode] is
  /// [TensorBufferMode.hostMemory]. [write] receives a [Float32List] view backed
  /// by the model-owned 64-byte-aligned host memory passed to
  /// `LiteRtCreateTensorBufferFromHostMemory`.
  ///
  /// Throws [StateError] while an async dispatch is in flight; the
  /// [dispatchAsync] future completes only after the run finishes, so
  /// sequential awaited `writeInput` → `dispatchAsync` → `readOutput`
  /// usage is safe.
  void writeInput(int index, void Function(Float32List input) write) {
    _ensureOpen();
    _ensureNoAsyncDispatch('writeInput');
    _ensureHostMemoryMode('writeInput');
    RangeError.checkValidIndex(index, this, 'index', _inputCount);
    final hostMemory = _inputHostMemory[index]!;
    write(hostMemory.asFloat32List(_inputByteSizes[index]));
  }

  /// Runs inference using inputs previously written with [writeInput].
  ///
  /// This is available only when [tensorBufferMode] is
  /// [TensorBufferMode.hostMemory].
  void dispatch() {
    _ensureOpen();
    _ensureNoAsyncDispatch('dispatch');
    _ensureHostMemoryMode('dispatch');
    _dispatch();
  }

  /// Runs inference asynchronously using inputs previously written with
  /// [writeInput].
  ///
  /// This is available only when [tensorBufferMode] is
  /// [TensorBufferMode.hostMemory]. The blocking native call runs on a
  /// helper isolate and the returned future completes after it finishes, so
  /// the calling isolate stays responsive meanwhile. Do not overlap calls:
  /// a dispatch while another is in flight throws [StateError].
  Future<void> dispatchAsync() async {
    _ensureOpen();
    _ensureHostMemoryMode('dispatchAsync');
    await _dispatchAsync();
  }

  /// Reads output [index] directly from this model's host-memory tensor buffer.
  ///
  /// This is available only when [tensorBufferMode] is
  /// [TensorBufferMode.hostMemory]. [read] receives a [Float32List] view backed
  /// by the model-owned host memory; use the callback to avoid allocating output
  /// copies in hot paths.
  R readOutput<R>(int index, R Function(Float32List output) read) {
    _ensureOpen();
    _ensureNoAsyncDispatch('readOutput');
    _ensureHostMemoryMode('readOutput');
    RangeError.checkValidIndex(index, this, 'index', _outputCount);
    final hostMemory = _outputHostMemory[index]!;
    return read(hostMemory.asFloat32List(_outputByteSizes[index]));
  }

  Future<void> _dispatchAsync() async {
    _ensureOpen();
    if (_dispatchInFlight) {
      throw StateError(
        'CompiledModel dispatch already in flight; await the previous '
        'runAsync/dispatchAsync call first.',
      );
    }
    _dispatchInFlight = true;
    try {
      final dispatcher = _asyncDispatcher ??= await _AsyncDispatcher.spawn();
      final error = await dispatcher.dispatch(
        compiledModelAddress: _compiledModel.address,
        inputCount: _inputCount,
        inputBuffersAddress: _inputBuffers.address,
        outputCount: _outputCount,
        outputBuffersAddress: _outputBuffers.address,
      );
      if (error != null) {
        throw StateError(error);
      }
    } finally {
      _dispatchInFlight = false;
    }
  }

  R _withLockedFloats<R>(
    Pointer<Void> buffer,
    int byteSize,
    int lockMode,
    String kind,
    int index,
    R Function(Float32List) fn,
  ) {
    final hostAddress = _lockScratch..value = nullptr;
    var locked = false;
    try {
      final lockStatus = _rt.lockTensorBuffer(buffer, hostAddress, lockMode);
      if (lockStatus != _kLiteRtStatusOk) {
        // A bare status code is not diagnosable: this failure has been seen
        // intermittently on Windows for a large output buffer, and the code
        // alone says nothing about which buffer, how big, or under which
        // accelerator. Build the context only on the failure path so the
        // happy path stays allocation-free.
        throw StateError(
          'LiteRtLockTensorBuffer $kind[$index] failed with '
          'LiteRtStatus=${describeLiteRtStatus(lockStatus)} '
          '(byteSize=$byteSize, lockMode=$lockMode, '
          'tensorBufferMode=${_tensorBufferMode.name}, '
          'accelerators={${_accelerators.map((a) => a.name).join(', ')}}).',
        );
      }
      locked = true;
      final bytes = hostAddress.value.cast<Uint8>().asTypedList(byteSize);
      final floats = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        byteSize ~/ sizeOf<Float>(),
      );
      return fn(floats);
    } finally {
      if (locked) {
        _checkAt(
          _rt.unlockTensorBuffer(buffer),
          'LiteRtUnlockTensorBuffer',
          kind,
          index,
        );
      }
    }
  }

  /// Releases native CompiledModel resources.
  ///
  /// Throws [StateError] when an async dispatch is still in flight: the
  /// helper isolate would be running the model against buffers this method
  /// frees. Await pending [runAsync]/[dispatchAsync] futures first.
  void close() {
    if (_closed) return;
    if (_dispatchInFlight) {
      throw StateError(
        'CompiledModel.close() called while an async dispatch is in flight.',
      );
    }
    _asyncDispatcher?.shutdown();
    _asyncDispatcher = null;
    calloc.free(_lockScratch);
    _releaseNative(
      _rt,
      inputBuffers: _inputBuffers,
      inputCount: _inputCount,
      outputBuffers: _outputBuffers,
      outputCount: _outputCount,
      compiledModel: _compiledModel,
      model: _model,
      modelBuffer: _modelBuffer,
      options: _options,
      environment: _environment,
      gpuOptionsIdentifier: _gpuOptionsIdentifier,
      hostMemoryAllocations: _hostMemoryAllocations,
    );
    _closed = true;
  }

  void _writeInput(int index, Float32List input) {
    final expectedFloats = _inputByteSizes[index] ~/ sizeOf<Float>();
    if (input.length != expectedFloats) {
      throw ArgumentError.value(
        input.length,
        'inputs[$index].length',
        'Expected $expectedFloats float32 values.',
      );
    }

    final hostMemory = _inputHostMemory[index];
    if (hostMemory != null) {
      hostMemory.asFloat32List(_inputByteSizes[index]).setAll(0, input);
      return;
    }

    _withLockedFloats(
      _inputBuffers[index],
      _inputByteSizes[index],
      _kLiteRtTensorBufferLockModeWrite,
      'input',
      index,
      (floats) => floats.setAll(0, input),
    );
  }

  Float32List _readOutput(int index) {
    final hostMemory = _outputHostMemory[index];
    if (hostMemory != null) {
      return Float32List.fromList(
        hostMemory.asFloat32List(_outputByteSizes[index]),
      );
    }

    return _withLockedFloats(
      _outputBuffers[index],
      _outputByteSizes[index],
      _kLiteRtTensorBufferLockModeRead,
      'output',
      index,
      Float32List.fromList,
    );
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('CompiledModel is already closed.');
    }
  }

  void _ensureNoAsyncDispatch(String method) {
    if (_dispatchInFlight) {
      throw StateError(
        'CompiledModel.$method called while an async dispatch is in flight; '
        'await the pending runAsync/dispatchAsync future first.',
      );
    }
  }

  void _ensureHostMemoryMode(String method) {
    if (_tensorBufferMode != TensorBufferMode.hostMemory) {
      throw StateError(
        'CompiledModel.$method requires '
        'TensorBufferMode.${TensorBufferMode.hostMemory.name}.',
      );
    }
  }
}

/// Runs blocking `LiteRtRunCompiledModel` calls on a dedicated helper
/// isolate so awaited dispatches leave the calling isolate's event loop
/// free.
///
/// The native handles are process-wide, so the helper drives the same
/// compiled model through raw addresses; its own isolate lazily reopens the
/// runtime library on first use. The owning [CompiledModel] serializes
/// dispatches, so the helper never runs the model concurrently.
final class _AsyncDispatcher {
  _AsyncDispatcher._(this._isolate, this._commands);

  final Isolate _isolate;
  final SendPort _commands;

  static Future<_AsyncDispatcher> spawn() async {
    final handshake = ReceivePort();
    final isolate = await Isolate.spawn(
      _dispatchLoop,
      handshake.sendPort,
      debugName: 'CompiledModel dispatcher',
    );
    final commands = await handshake.first as SendPort;
    return _AsyncDispatcher._(isolate, commands);
  }

  /// Runs one dispatch; completes with null on success or an error message.
  Future<String?> dispatch({
    required int compiledModelAddress,
    required int inputCount,
    required int inputBuffersAddress,
    required int outputCount,
    required int outputBuffersAddress,
  }) async {
    final reply = ReceivePort();
    _commands.send([
      reply.sendPort,
      compiledModelAddress,
      inputCount,
      inputBuffersAddress,
      outputCount,
      outputBuffersAddress,
    ]);
    final result = await reply.first;
    reply.close();
    return result as String?;
  }

  void shutdown() => _isolate.kill(priority: Isolate.immediate);

  static void _dispatchLoop(SendPort handshake) {
    final commands = ReceivePort();
    handshake.send(commands.sendPort);
    commands.listen((message) {
      final args = message as List<Object?>;
      final reply = args[0] as SendPort;
      String? error;
      try {
        final status = litert.runCompiledModel(
          Pointer<Void>.fromAddress(args[1] as int),
          0,
          args[2] as int,
          Pointer<Pointer<Void>>.fromAddress(args[3] as int),
          args[4] as int,
          Pointer<Pointer<Void>>.fromAddress(args[5] as int),
        );
        if (status != _kLiteRtStatusOk) {
          error =
              'LiteRtRunCompiledModel failed with LiteRtStatus='
              '${describeLiteRtStatus(status)}.';
        }
      } catch (e) {
        error = 'CompiledModel async dispatch failed: $e';
      }
      reply.send(error);
    });
  }
}

final class _HostMemoryAllocation {
  _HostMemoryAllocation._(this.raw, this.aligned);

  factory _HostMemoryAllocation.allocate(int byteSize) {
    // calloc (zero-initialized): the official docs require delegate padding in
    // the buffer to be "included and initialized".
    final raw = calloc<Uint8>(byteSize + _kHostMemoryAlignment - 1);
    final alignedAddress =
        (raw.address + _kHostMemoryAlignment - 1) &
        ~(_kHostMemoryAlignment - 1);
    return _HostMemoryAllocation._(
      raw,
      Pointer<Uint8>.fromAddress(alignedAddress),
    );
  }

  final Pointer<Uint8> raw;
  final Pointer<Uint8> aligned;

  Float32List asFloat32List(int byteSize) =>
      aligned.cast<Float>().asTypedList(byteSize ~/ sizeOf<Float>());

  void free() => calloc.free(raw);
}

Pointer<Void> _createEnvironment(LiteRtBindings rt) {
  final runtimeDir = litertRuntimeDir;
  final out = calloc<Pointer<Void>>();
  Pointer<LiteRtEnvOption>? envOptions;
  Pointer<Utf8>? runtimeDirPtr;
  try {
    if (runtimeDir == null) {
      _check(
        'LiteRtCreateEnvironment',
        rt.createEnvironment(0, nullptr.cast<LiteRtEnvOption>(), out),
      );
    } else {
      envOptions = calloc<LiteRtEnvOption>();
      runtimeDirPtr = runtimeDir.toNativeUtf8();
      envOptions.ref
        ..tag = kLiteRtEnvOptionTagRuntimeLibraryDir
        ..value.type = kLiteRtAnyTypeString
        ..value.value.strValue = runtimeDirPtr;
      _check(
        'LiteRtCreateEnvironment',
        rt.createEnvironment(1, envOptions, out),
      );
    }
    return out.value;
  } finally {
    if (envOptions != null) {
      calloc.free(envOptions);
    }
    if (runtimeDirPtr != null) {
      malloc.free(runtimeDirPtr);
    }
    calloc.free(out);
  }
}

Pointer<Void> _createOptions(LiteRtBindings rt, int acceleratorMask) {
  final out = calloc<Pointer<Void>>();
  try {
    _check('LiteRtCreateOptions', rt.createOptions(out));
    final options = out.value;
    _check(
      'LiteRtSetOptionsHardwareAccelerators',
      rt.setOptionsHardwareAccelerators(options, acceleratorMask),
    );
    return options;
  } catch (_) {
    if (out.value != nullptr) {
      rt.destroyOptions(out.value);
    }
    rethrow;
  } finally {
    calloc.free(out);
  }
}

Pointer<Utf8> _addGpuFp32Options(LiteRtBindings rt, Pointer<Void> options) {
  final identifier = 'gpu_options'.toNativeUtf8();
  final payload = 'precision = $_kLiteRtDelegatePrecisionFp32\n'.toNativeUtf8();
  final opaqueOut = calloc<Pointer<Void>>();
  var payloadTransferred = false;
  var optionsTransferred = false;

  try {
    _check(
      'LiteRtCreateOpaqueOptions',
      rt.createOpaqueOptions(
        identifier,
        payload.cast<Void>(),
        malloc.nativeFree
            .cast<NativeFunction<LiteRtOpaquePayloadDeleterNative>>(),
        opaqueOut,
      ),
    );
    payloadTransferred = true;
    _check(
      'LiteRtAddOpaqueOptions',
      rt.addOpaqueOptions(options, opaqueOut.value),
    );
    optionsTransferred = true;
    return identifier;
  } finally {
    if (!optionsTransferred && opaqueOut.value != nullptr) {
      rt.destroyOpaqueOptions(opaqueOut.value);
    }
    if (!payloadTransferred) {
      malloc.free(payload);
    }
    calloc.free(opaqueOut);
  }
}

final class _ModelSource {
  const _ModelSource(this.model, [this.modelBuffer]);

  final Pointer<Void> model;
  final Pointer<Uint8>? modelBuffer;
}

_ModelSource _createModelFromFile(LiteRtBindings rt, String path) {
  final pathPtr = path.toNativeUtf8();
  final out = calloc<Pointer<Void>>();
  try {
    _check('LiteRtCreateModelFromFile', rt.createModelFromFile(pathPtr, out));
    return _ModelSource(out.value);
  } finally {
    malloc.free(pathPtr);
    calloc.free(out);
  }
}

_ModelSource _createModelFromBuffer(LiteRtBindings rt, Uint8List bytes) {
  if (bytes.isEmpty) {
    throw ArgumentError.value(
      bytes.length,
      'bytes.length',
      'Must be non-zero.',
    );
  }

  final buffer = malloc<Uint8>(bytes.length);
  buffer.asTypedList(bytes.length).setAll(0, bytes);
  final out = calloc<Pointer<Void>>();

  try {
    _check(
      'LiteRtCreateModelFromBuffer',
      rt.createModelFromBuffer(buffer.cast<Void>(), bytes.length, out),
    );
    return _ModelSource(out.value, buffer);
  } catch (_) {
    malloc.free(buffer);
    rethrow;
  } finally {
    calloc.free(out);
  }
}

Pointer<Void> _createCompiledModel(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> model,
  Pointer<Void> options,
) {
  final out = calloc<Pointer<Void>>();
  try {
    _check(
      'LiteRtCreateCompiledModel',
      rt.createCompiledModel(environment, model, options, out),
    );
    return out.value;
  } finally {
    calloc.free(out);
  }
}

Pointer<Void> _getModelSignature(LiteRtBindings rt, Pointer<Void> model) {
  final out = calloc<Pointer<Void>>();
  try {
    _check('LiteRtGetModelSignature', rt.getModelSignature(model, 0, out));
    return out.value;
  } finally {
    calloc.free(out);
  }
}

int _getSignatureInputCount(LiteRtBindings rt, Pointer<Void> signature) {
  final out = calloc<IntPtr>();
  try {
    _check(
      'LiteRtGetNumSignatureInputs',
      rt.getNumSignatureInputs(signature, out),
    );
    return out.value;
  } finally {
    calloc.free(out);
  }
}

int _getSignatureOutputCount(LiteRtBindings rt, Pointer<Void> signature) {
  final out = calloc<IntPtr>();
  try {
    _check(
      'LiteRtGetNumSignatureOutputs',
      rt.getNumSignatureOutputs(signature, out),
    );
    return out.value;
  } finally {
    calloc.free(out);
  }
}

Pointer<Void> _createInputBuffer(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> compiledModel,
  Pointer<Void> signature,
  int index,
  List<int> inputByteSizes,
  TensorBufferMode tensorBufferMode,
  List<_HostMemoryAllocation?> inputHostMemory,
  List<_HostMemoryAllocation> hostMemoryAllocations,
) {
  return _withRankedTensorType(
    rt,
    'input[$index]',
    signature,
    index,
    isInput: true,
    (tensorType) {
      final layoutPtr =
          (tensorType.cast<Uint8>() + kLiteRtRankedTensorTypeLayoutOffset)
              .cast<LiteRtLayout>();
      _check(
        'LiteRtGetCompiledModelInputTensorLayout input[$index]',
        rt.getCompiledModelInputTensorLayout(
          compiledModel,
          0,
          index,
          layoutPtr,
        ),
      );
      return _createBufferFromRequirements(
        rt,
        environment,
        compiledModel,
        index,
        tensorType,
        inputByteSizes,
        tensorBufferMode: tensorBufferMode,
        hostMemoryForTensor: inputHostMemory,
        hostMemoryAllocations: hostMemoryAllocations,
        isInput: true,
      );
    },
  );
}

Pointer<Void> _createOutputBuffer(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> compiledModel,
  Pointer<Void> signature,
  int index,
  Pointer<LiteRtLayout> outputLayout,
  List<int> outputByteSizes,
  TensorBufferMode tensorBufferMode,
  List<_HostMemoryAllocation?> outputHostMemory,
  List<_HostMemoryAllocation> hostMemoryAllocations,
) {
  return _withRankedTensorType(
    rt,
    'output[$index]',
    signature,
    index,
    isInput: false,
    (tensorType) {
      (tensorType.cast<Uint8>() + kLiteRtRankedTensorTypeLayoutOffset)
          .asTypedList(kLiteRtLayoutByteSize)
          .setAll(
            0,
            outputLayout.cast<Uint8>().asTypedList(kLiteRtLayoutByteSize),
          );
      return _createBufferFromRequirements(
        rt,
        environment,
        compiledModel,
        index,
        tensorType,
        outputByteSizes,
        tensorBufferMode: tensorBufferMode,
        hostMemoryForTensor: outputHostMemory,
        hostMemoryAllocations: hostMemoryAllocations,
        isInput: false,
      );
    },
  );
}

Pointer<Void> _withRankedTensorType(
  LiteRtBindings rt,
  String label,
  Pointer<Void> signature,
  int index,
  Pointer<Void> Function(Pointer<LiteRtRankedTensorType>) action, {
  required bool isInput,
}) {
  final tensorOut = calloc<Pointer<Void>>();
  final tensorType = calloc<Uint8>(
    kLiteRtRankedTensorTypeByteSize,
  ).cast<LiteRtRankedTensorType>();
  try {
    final tensorStatus = isInput
        ? rt.getSignatureInputTensorByIndex(signature, index, tensorOut)
        : rt.getSignatureOutputTensorByIndex(signature, index, tensorOut);
    _check('LiteRtGetSignatureTensorByIndex $label', tensorStatus);
    _check(
      'LiteRtGetRankedTensorType $label',
      rt.getRankedTensorType(tensorOut.value, tensorType),
    );
    return action(tensorType);
  } finally {
    calloc.free(tensorType);
    calloc.free(tensorOut);
  }
}

Pointer<Void> _createBufferFromRequirements(
  LiteRtBindings rt,
  Pointer<Void> environment,
  Pointer<Void> compiledModel,
  int index,
  Pointer<LiteRtRankedTensorType> tensorType,
  List<int> byteSizes, {
  required TensorBufferMode tensorBufferMode,
  required List<_HostMemoryAllocation?> hostMemoryForTensor,
  required List<_HostMemoryAllocation> hostMemoryAllocations,
  required bool isInput,
}) {
  final requirementsOut = calloc<Pointer<Void>>();
  final sizeOut = calloc<IntPtr>();
  final bufferOut = calloc<Pointer<Void>>();
  final label = isInput ? 'input[$index]' : 'output[$index]';
  _HostMemoryAllocation? hostMemory;

  try {
    final requirementsStatus = isInput
        ? rt.getCompiledModelInputBufferRequirements(
            compiledModel,
            0,
            index,
            requirementsOut,
          )
        : rt.getCompiledModelOutputBufferRequirements(
            compiledModel,
            0,
            index,
            requirementsOut,
          );
    _check(
      'LiteRtGetCompiledModelBufferRequirements $label',
      requirementsStatus,
    );
    _check(
      'LiteRtGetTensorBufferRequirementsBufferSize $label',
      rt.getTensorBufferRequirementsBufferSize(requirementsOut.value, sizeOut),
    );
    _checkFloat32Tensor(label, tensorType, sizeOut.value);
    switch (tensorBufferMode) {
      case TensorBufferMode.managed:
        _check(
          'LiteRtCreateManagedTensorBufferFromRequirements $label',
          rt.createManagedTensorBufferFromRequirements(
            environment,
            tensorType,
            requirementsOut.value,
            bufferOut,
          ),
        );
        hostMemoryForTensor.add(null);
      case TensorBufferMode.hostMemory:
        hostMemory = _HostMemoryAllocation.allocate(sizeOut.value);
        _check(
          'LiteRtCreateTensorBufferFromHostMemory $label',
          rt.createTensorBufferFromHostMemory(
            tensorType,
            hostMemory.aligned.cast<Void>(),
            sizeOut.value,
            nullptr,
            bufferOut,
          ),
        );
        hostMemoryForTensor.add(hostMemory);
        hostMemoryAllocations.add(hostMemory);
        hostMemory = null;
    }
    byteSizes.add(sizeOut.value);
    return bufferOut.value;
  } catch (_) {
    hostMemory?.free();
    rethrow;
  } finally {
    // The requirements handle is borrowed from LiteRtCompiledModel; only the
    // holder pointer is owned here.
    calloc.free(bufferOut);
    calloc.free(sizeOut);
    calloc.free(requirementsOut);
  }
}

void _releaseNative(
  LiteRtBindings rt, {
  Pointer<Pointer<Void>>? inputBuffers,
  int inputCount = 0,
  Pointer<Pointer<Void>>? outputBuffers,
  int outputCount = 0,
  Pointer<Void>? compiledModel,
  Pointer<Void>? model,
  Pointer<Uint8>? modelBuffer,
  Pointer<Void>? options,
  Pointer<Void>? environment,
  Pointer<Utf8>? gpuOptionsIdentifier,
  List<_HostMemoryAllocation>? hostMemoryAllocations,
}) {
  if (outputBuffers != null) {
    for (var i = 0; i < outputCount; i++) {
      if (outputBuffers[i] != nullptr) {
        rt.destroyTensorBuffer(outputBuffers[i]);
      }
    }
    calloc.free(outputBuffers);
  }
  if (inputBuffers != null) {
    for (var i = 0; i < inputCount; i++) {
      if (inputBuffers[i] != nullptr) {
        rt.destroyTensorBuffer(inputBuffers[i]);
      }
    }
    calloc.free(inputBuffers);
  }
  if (hostMemoryAllocations != null) {
    for (final allocation in hostMemoryAllocations) {
      allocation.free();
    }
  }
  if (compiledModel != null && compiledModel != nullptr) {
    rt.destroyCompiledModel(compiledModel);
  }
  if (model != null && model != nullptr) {
    rt.destroyModel(model);
  }
  if (modelBuffer != null && modelBuffer != nullptr) {
    malloc.free(modelBuffer);
  }
  if (options != null && options != nullptr) {
    rt.destroyOptions(options);
  }
  // The environment is a per-isolate shared singleton
  // (CompiledModel._sharedEnvironment) reused across every CompiledModel, so it
  // is intentionally never destroyed here; it lives for the isolate's lifetime.
  if (environment != null &&
      environment != nullptr &&
      environment != CompiledModel._sharedEnvironment) {
    rt.destroyEnvironment(environment);
  }
  if (gpuOptionsIdentifier != null && gpuOptionsIdentifier != nullptr) {
    malloc.free(gpuOptionsIdentifier);
  }
}

int _acceleratorMask(Set<Accelerator> accelerators) {
  if (accelerators.isEmpty) {
    throw ArgumentError.value(
      accelerators,
      'accelerators',
      'Must contain at least one accelerator.',
    );
  }

  return accelerators.fold(0, (mask, accelerator) {
    final value = switch (accelerator) {
      Accelerator.cpu => _kLiteRtHwAcceleratorCpu,
      Accelerator.gpu => _kLiteRtHwAcceleratorGpu,
      Accelerator.npu => _kLiteRtHwAcceleratorNpu,
    };
    return mask | value;
  });
}

void _checkFloat32Tensor(
  String label,
  Pointer<LiteRtRankedTensorType> tensorType,
  int byteSize,
) {
  // element_type is an int32 at offset 0 on every supported ABI.
  final elementType = tensorType.cast<Int32>().value;
  if (elementType != _kLiteRtElementTypeFloat32) {
    throw UnsupportedError(
      'CompiledModel.run supports Float32 tensors only; $label has LiteRT '
      'element type $elementType.',
    );
  }
  if (byteSize % sizeOf<Float>() != 0) {
    throw StateError('$label byte size $byteSize is not float32-aligned.');
  }
}

void _checkStructLayouts() {
  // LiteRtLayout/LiteRtRankedTensorType are opaque on the Dart side because
  // their ABI differs per compiler (MSVC vs clang/gcc bitfield packing); both
  // layouts are pinned by static_asserts in litert/c/litert_layout.h and
  // sized here via kLiteRtLayoutByteSize/kLiteRtRankedTensorTypeByteSize.
  if (sizeOf<LiteRtAny>() != _kLiteRtAnySize) {
    throw StateError(
      'LiteRtAny size ${sizeOf<LiteRtAny>()} != $_kLiteRtAnySize.',
    );
  }
  if (sizeOf<LiteRtEnvOption>() != _kLiteRtEnvOptionSize) {
    throw StateError(
      'LiteRtEnvOption size ${sizeOf<LiteRtEnvOption>()} != '
      '$_kLiteRtEnvOptionSize.',
    );
  }
  _checkLiteRtAnyValueOffset();
  _checkLiteRtEnvOptionValueOffset();
}

void _checkLiteRtAnyValueOffset() {
  final any = calloc<LiteRtAny>();
  try {
    any.ref.value.intValue = 0x0102030405060708;
    final bytes = any.cast<Uint8>().asTypedList(_kLiteRtAnySize);
    final value = ByteData.sublistView(
      bytes,
    ).getInt64(_kLiteRtAnyValueOffset, Endian.host);
    if (value != 0x0102030405060708) {
      throw StateError(
        'LiteRtAny union offset does not match $_kLiteRtAnyValueOffset.',
      );
    }
  } finally {
    calloc.free(any);
  }
}

void _checkLiteRtEnvOptionValueOffset() {
  final option = calloc<LiteRtEnvOption>();
  try {
    option.ref.value.type = 0x01020304;
    final bytes = option.cast<Uint8>().asTypedList(_kLiteRtEnvOptionSize);
    final value = ByteData.sublistView(
      bytes,
    ).getInt32(_kLiteRtEnvOptionValueOffset, Endian.host);
    if (value != 0x01020304) {
      throw StateError(
        'LiteRtEnvOption.value offset does not match '
        '$_kLiteRtEnvOptionValueOffset.',
      );
    }
  } finally {
    calloc.free(option);
  }
}

void _check(String operation, int status) {
  if (status != _kLiteRtStatusOk) {
    throw StateError(
      '$operation failed with LiteRtStatus=${describeLiteRtStatus(status)}.',
    );
  }
}

/// Like [_check] for per-tensor operations, taking the label pieces so the
/// happy path never builds an interpolated string. Callers pass const
/// [operation] and [kind]; the full label exists only when throwing.
void _checkAt(int status, String operation, String kind, int index) {
  if (status != _kLiteRtStatusOk) {
    throw StateError(
      '$operation $kind[$index] failed with '
      'LiteRtStatus=${describeLiteRtStatus(status)}.',
    );
  }
}
