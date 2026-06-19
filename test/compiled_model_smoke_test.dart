@TestOn('mac-os || linux || windows')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Exercises the real package CompiledModel path end-to-end:
/// loads libLiteRt via the package loader, compiles a real .tflite, sets up
/// managed tensor buffers, and frees them. Requires the LiteRT Next runtime.
/// On macOS it is bundled with the package; on Linux/Windows point the loader
/// at an extracted `ai-edge-litert` wheel library:
///   LITERT_LIB_PATH=/path/to/libLiteRt.so flutter test test/compiled_model_smoke_test.dart
void main() {
  const model = 'example/assets/simple_model.tflite';

  test('CompiledModel compiles a model on the CPU accelerator', () {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);
    expect(cm, isNotNull);
    expect(cm.accelerators, {Accelerator.cpu});
  });

  test('CompiledModel compiles model bytes on the CPU accelerator', () {
    final bytes = File(model).readAsBytesSync();
    final cm = CompiledModel.fromBuffer(bytes, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);
    expect(cm, isNotNull);
  });

  // NOTE: The GPU (Metal) accelerator needs a real GPU/Metal device context,
  // which the headless `flutter test` runner does not provide; there it fails
  // with kLiteRtStatusErrorCompilation (504) because the Metal accelerator can't
  // register. The GPU path is validated separately in a real `flutter run`
  // app (which has Metal access). These tests assert success when Metal is
  // available and skip gracefully when it isn't.
  test(
    'CompiledModel compiles on the GPU (Metal) accelerator when available',
    () {
      _gpuTestOrSkip(
        () => CompiledModel.fromFile(model, accelerators: {Accelerator.gpu}),
      );
    },
  );

  // Asserts the GPU→CPU fallback path builds a usable model where a GPU
  // accelerator can register (macOS Metal, Windows). On a headless Linux
  // runner the bundled WebGPU accelerator finds no adapter and the create
  // fails outright rather than falling back, so this skips there, same as
  // the strict-GPU tests above.
  test('CompiledModel compiles with GPU and CPU fallback', () {
    _gpuTestOrSkip(
      () => CompiledModel.fromFile(
        model,
        accelerators: {Accelerator.gpu, Accelerator.cpu},
      ),
    );
  });

  test('run() produces correct results on the CPU accelerator', () async {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    final input = Float32List(cm.inputByteSizes[0] ~/ 4);
    input[0] = 3.0;

    final syncOut = cm.run([input]);
    final asyncOut = await cm.runAsync([input]);

    expect(asyncOut, syncOut);
    // simple_model is y = 2x + 1.
    expect(syncOut[0][0], closeTo(2 * input[0] + 1, 1e-5));
  });

  test('host-memory tensor buffers match managed buffers on CPU', () async {
    final managed = CompiledModel.fromFile(
      model,
      accelerators: {Accelerator.cpu},
    );
    final hostMemory = CompiledModel.fromFile(
      model,
      accelerators: {Accelerator.cpu},
      tensorBufferMode: TensorBufferMode.hostMemory,
    );
    addTearDown(managed.close);
    addTearDown(hostMemory.close);

    expect(managed.tensorBufferMode, TensorBufferMode.managed);
    expect(hostMemory.tensorBufferMode, TensorBufferMode.hostMemory);
    expect(hostMemory.inputByteSizes, managed.inputByteSizes);
    expect(hostMemory.outputByteSizes, managed.outputByteSizes);

    final input = Float32List(managed.inputByteSizes[0] ~/ 4);
    for (var i = 0; i < input.length; i++) {
      input[i] = i + 3.0;
    }

    final managedOut = managed.run([input]);
    final hostSyncOut = hostMemory.run([input]);
    final hostAsyncOut = await hostMemory.runAsync([input]);

    expect(hostSyncOut, managedOut);
    expect(hostAsyncOut, managedOut);
    // simple_model is y = 2x + 1.
    expect(hostSyncOut[0][0], closeTo(2 * input[0] + 1, 1e-5));
  });

  test('host-memory direct write/dispatch/read matches run() on CPU', () async {
    final managed = CompiledModel.fromFile(
      model,
      accelerators: {Accelerator.cpu},
    );
    final hostMemory = CompiledModel.fromFile(
      model,
      accelerators: {Accelerator.cpu},
      tensorBufferMode: TensorBufferMode.hostMemory,
    );
    addTearDown(managed.close);
    addTearDown(hostMemory.close);

    final input = Float32List(managed.inputByteSizes[0] ~/ 4);
    for (var i = 0; i < input.length; i++) {
      input[i] = i + 5.0;
    }
    final managedOut = managed.run([input]);

    hostMemory.writeInput(0, (view) {
      expect(view.length, input.length);
      view.setAll(0, input);
    });
    hostMemory.dispatch();
    final directSyncOut = List<Float32List>.generate(
      hostMemory.outputCount,
      (i) => hostMemory.readOutput(i, Float32List.fromList),
    );

    hostMemory.writeInput(0, (view) => view.setAll(0, input));
    await hostMemory.dispatchAsync();
    final directAsyncOut = List<Float32List>.generate(
      hostMemory.outputCount,
      (i) => hostMemory.readOutput(i, Float32List.fromList),
    );

    expect(directSyncOut, managedOut);
    expect(directAsyncOut, managedOut);
    // simple_model is y = 2x + 1.
    expect(directAsyncOut[0][0], closeTo(2 * input[0] + 1, 1e-5));
  });

  test('direct host-memory API rejects managed tensor buffers', () async {
    final cm = CompiledModel.fromFile(model, accelerators: {Accelerator.cpu});
    addTearDown(cm.close);

    expect(() => cm.writeInput(0, (_) {}), throwsStateError);
    expect(cm.dispatch, throwsStateError);
    await expectLater(cm.dispatchAsync(), throwsStateError);
    expect(() => cm.readOutput(0, (view) => view.length), throwsStateError);
  });

  test(
    'host-memory tensor buffers match managed buffers on GPU when available',
    () async {
      CompiledModel? managed;
      CompiledModel? hostMemory;
      try {
        managed = CompiledModel.fromFile(
          model,
          accelerators: {Accelerator.gpu},
        );
        hostMemory = CompiledModel.fromFile(
          model,
          accelerators: {Accelerator.gpu},
          tensorBufferMode: TensorBufferMode.hostMemory,
        );
      } on StateError catch (e) {
        managed?.close();
        hostMemory?.close();
        if (_isGpuUnavailable(e)) {
          markTestSkipped(
            'GPU accelerator unavailable in headless test runner.',
          );
          return;
        }
        rethrow;
      }
      final managedModel = managed;
      final hostMemoryModel = hostMemory;
      addTearDown(managedModel.close);
      addTearDown(hostMemoryModel.close);

      final input = Float32List(managedModel.inputByteSizes[0] ~/ 4);
      input[0] = 3.0;

      final managedOut = await managedModel.runAsync([input]);
      final hostOut = await hostMemoryModel.runAsync([input]);

      expect(hostOut, managedOut);
      // simple_model is y = 2x + 1.
      expect(hostOut[0][0], closeTo(2 * input[0] + 1, 1e-5));
    },
  );

  test('CompiledModel GPU FP32 precision option compiles when available', () {
    _gpuTestOrSkip(
      () => CompiledModel.fromFile(
        model,
        accelerators: {Accelerator.gpu},
        precision: Precision.fp32,
      ),
    );
  });
}

void _gpuTestOrSkip(CompiledModel Function() build) {
  CompiledModel cm;
  try {
    cm = build();
  } on StateError catch (e) {
    if (_isGpuUnavailable(e)) {
      markTestSkipped('GPU accelerator unavailable in headless test runner.');
      return;
    }
    rethrow;
  }
  addTearDown(cm.close);
  expect(cm, isNotNull);
}

/// Whether [e] means "the GPU accelerator can't run here", as opposed to a
/// real failure.
///
/// On macOS the Metal accelerator is always bundled, so only the documented
/// headless-compilation status (504) is acceptable. On Linux the WebGPU
/// accelerator is now bundled, but headless CI runners expose no adapter
/// ("Found 0 adapters"), so the create fails with a runtime status (3); on
/// Windows only the runtime library is staged. On either, any LiteRT status
/// from a GPU-requesting create means the accelerator is unavailable in CI.
bool _isGpuUnavailable(StateError e) {
  if (e.message.contains('LiteRtStatus=504')) return true;
  return (Platform.isLinux || Platform.isWindows) &&
      e.message.contains('LiteRtStatus=');
}
