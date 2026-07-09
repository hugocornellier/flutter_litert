@TestOn('browser')
library;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
// The analyzer resolves the conditional export to the platform-neutral stub,
// which does not carry the web-only watchdog seam, so the watchdog group
// addresses the web implementation directly. At runtime on the browser
// platform this is the same class the public import resolves to.
import 'package:flutter_litert/src/compiled_model/compiled_model_web.dart'
    as cm_web;
import 'package:flutter_test/flutter_test.dart';

/// `test/assets/add.tflite` (a float32 graph computing x + x + x on a
/// [1, 8, 8, 3] tensor), embedded because browser tests cannot read the
/// file system.
const _addModelBase64 =
    'JAAAAFRGTDMAAAAAAAAAABQAGAAEAAgADAAAABAAAAAAABQAFAAAAAMAAADkAQAAmAAAAIAAAAAE'
    'AAAAAQAAABAAAAAAAAoAEAAEAAgADAAKAAAAPAAAABwAAAAEAAAADwAAAHNlcnZpbmdfZGVmYXVs'
    'dAABAAAABAAAAOT///8IAAAAAgAAAAEAAAB4AAAAAQAAAAwAAAAIAAwABAAIAAgAAAAIAAAAAQAA'
    'AAEAAABhAAAAAQAAAAQAAACk/v//AAAAAAAAAAABAAAAEAAAAAwAFAAEAAgADAAQAAwAAACUAAAA'
    'iAAAAHwAAAAEAAAAAgAAAEQAAAAEAAAA0v///wAAAAsYAAAADAAAAAQAAAD4/v//AQAAAAIAAAAC'
    'AAAAAAAAAAEAAAAAAA4AFAAAAAgADAAHABAADgAAAAAAAAsYAAAADAAAAAQAAAA0////AQAAAAAA'
    'AAACAAAAAQAAAAEAAAABAAAAAgAAAAEAAAABAAAAAwAAAHAAAAA0AAAABAAAAKj///8UAAAABAAA'
    'AAYAAABvdXRwdXQAAAQAAAABAAAACAAAAAgAAAADAAAA1P///xQAAAAEAAAABQAAAGlucHV0AAAA'
    'BAAAAAEAAAAIAAAACAAAAAMAAAAMAAwABAAAAAAACAAMAAAAEAAAAAQAAAADAAAAYWRkAAQAAAAB'
    'AAAACAAAAAgAAAADAAAAAQAAAAgAAAAEAAQABAAAAA==';

void main() {
  final Uint8List modelBytes = base64Decode(_addModelBase64);

  group('web CompiledModel sync API', () {
    test('sync factories throw UnsupportedError', () {
      expect(() => CompiledModel.fromFile('m.tflite'), throwsUnsupportedError);
      expect(
        () => CompiledModel.fromBuffer(modelBytes),
        throwsUnsupportedError,
      );
      expect(
        () => CompiledModel.fromBufferWithGpuFallback(modelBytes),
        throwsUnsupportedError,
      );
    });

    test('fromBufferAsync validates arguments before loading the runtime', () {
      expect(
        () => CompiledModel.fromBufferAsync(modelBytes, accelerators: {}),
        throwsArgumentError,
      );
      expect(
        () => CompiledModel.fromBufferAsync(
          modelBytes,
          accelerators: {Accelerator.npu},
        ),
        throwsArgumentError,
      );
      expect(
        () => CompiledModel.fromBufferAsync(
          modelBytes,
          tensorBufferMode: TensorBufferMode.hostMemory,
        ),
        throwsUnsupportedError,
      );
      expect(
        () => CompiledModel.fromBufferAsync(Uint8List(0)),
        throwsArgumentError,
      );
    });
  });

  // These tests exercise the real LiteRT.js runtime, auto-loaded from
  // cdn.jsdelivr.net by the first fromBufferAsync call; they need network
  // access.
  group('web CompiledModel with LiteRT.js (network)', () {
    test('compiles on WASM, runs, and reports geometry', () async {
      final cm = await CompiledModel.fromBufferAsync(modelBytes);
      expect(cm.accelerators, {Accelerator.cpu});
      expect(cm.tensorBufferMode, TensorBufferMode.managed);
      expect(cm.inputCount, 1);
      expect(cm.outputCount, 1);
      expect(cm.inputByteSizes, [8 * 8 * 3 * 4]);
      expect(cm.outputByteSizes, [8 * 8 * 3 * 4]);

      final len = cm.inputByteSizes[0] ~/ 4;
      final input = Float32List(len)..fillRange(0, len, 1.0);
      final outputs = await cm.runAsync([input]);
      expect(outputs, hasLength(1));
      expect(outputs[0], hasLength(len));
      // add.tflite computes x + x + x.
      expect(outputs[0][0], closeTo(3.0, 1e-6));
      expect(outputs[0][len - 1], closeTo(3.0, 1e-6));
      cm.close();
    });

    test('queued runAsync calls serialize and both complete', () async {
      final cm = await CompiledModel.fromBufferAsync(modelBytes);
      final len = cm.inputByteSizes[0] ~/ 4;
      final a = Float32List(len)..fillRange(0, len, 1.0);
      final b = Float32List(len)..fillRange(0, len, 2.0);

      final results = await Future.wait([
        cm.runAsync([a]),
        cm.runAsync([b]),
      ]);
      expect(results[0][0][0], closeTo(3.0, 1e-6));
      expect(results[1][0][0], closeTo(6.0, 1e-6));
      cm.close();
    });

    test('guards match the native implementation', () async {
      final cm = await CompiledModel.fromBufferAsync(modelBytes);
      final len = cm.inputByteSizes[0] ~/ 4;
      final input = Float32List(len);

      // Sync inference is unimplementable on the web.
      expect(() => cm.run([input]), throwsUnsupportedError);

      // Host-memory APIs require a mode the web cannot provide.
      expect(() => cm.writeInput(0, (i) {}), throwsStateError);
      expect(() => cm.dispatch(), throwsStateError);
      expect(() => cm.dispatchAsync(), throwsStateError);
      expect(() => cm.readOutput(0, (o) => o[0]), throwsStateError);

      // Input count / length validation.
      expect(() => cm.runAsync([input, input]), throwsArgumentError);
      expect(
        () => cm.runAsync([Float32List(len - 1)]),
        throwsA(isA<ArgumentError>()),
      );

      // close() refuses while a dispatch is in flight, then works after.
      final pending = cm.runAsync([input]);
      await null;
      expect(cm.close, throwsStateError);
      await pending;
      cm.close();
      cm.close(); // Idempotent.
      expect(() => cm.runAsync([input]), throwsStateError);
    });

    test('compile failure surfaces as StateError', () async {
      final garbage = Uint8List.fromList([1, 2, 3, 4]);
      await expectLater(
        CompiledModel.fromBufferAsync(garbage),
        throwsStateError,
      );
    });

    test(
      'fromBufferWithGpuFallbackAsync always yields a working model',
      () async {
        Object? fallbackError;
        final cm = await CompiledModel.fromBufferWithGpuFallbackAsync(
          modelBytes,
          onFallback: (e) => fallbackError = e,
        );
        // Headless Chrome may or may not expose WebGPU; any resolution is
        // valid ({gpu, cpu} means partial WebGPU acceleration), but the
        // fallback callback must fire iff the WebGPU compile attempt failed.
        expect(
          cm.accelerators,
          anyOf(
            equals({Accelerator.gpu}),
            equals({Accelerator.cpu}),
            equals({Accelerator.gpu, Accelerator.cpu}),
          ),
        );
        expect(
          fallbackError,
          cm.accelerators.contains(Accelerator.gpu) ? isNull : isNotNull,
        );

        final len = cm.inputByteSizes[0] ~/ 4;
        final input = Float32List(len)..fillRange(0, len, 1.0);
        final outputs = await cm.runAsync([input]);
        expect(outputs[0][0], closeTo(3.0, 1e-6));
        cm.close();
      },
    );

    test('forceCpu skips the WebGPU attempt', () async {
      final cm = await CompiledModel.fromBufferWithGpuFallbackAsync(
        modelBytes,
        forceCpu: true,
      );
      expect(cm.accelerators, {Accelerator.cpu});
      cm.close();
    });
  });

  // LiteRT.js 2.4.0's compile promise can, very rarely, neither resolve nor
  // reject on GPU-less machines. These tests stand in a fake `window.LiteRt`
  // whose WebGPU compile hangs on purpose (WASM compiles delegate to the
  // real, network-loaded runtime) and assert the watchdog semantics: bounded
  // fallback where one exists, untouched strict requests, no leaked model.
  group('web CompiledModel WebGPU compile watchdog (network)', () {
    late JSObject realRoot;
    final Duration savedWatchdog = cm_web.CompiledModel.debugGpuCompileWatchdog;

    setUpAll(() async {
      // Load the real runtime once so the stub can delegate WASM compiles.
      (await cm_web.CompiledModel.fromBufferAsync(modelBytes)).close();
      realRoot = globalContext.getProperty<JSObject>('LiteRt'.toJS);
    });

    setUp(() {
      cm_web.CompiledModel.debugGpuCompileWatchdog = const Duration(
        milliseconds: 400,
      );
    });

    tearDown(() {
      cm_web.CompiledModel.debugGpuCompileWatchdog = savedWatchdog;
      globalContext.setProperty('LiteRt'.toJS, realRoot);
    });

    /// Replaces `window.LiteRt` with a stub whose webgpu `loadAndCompile`
    /// returns a promise that never settles on its own; wasm compiles
    /// delegate to the real runtime. The returned function settles the hung
    /// promise with the given model object.
    void Function(JSObject model) installHangingGpuRuntime() {
      JSFunction? gpuResolve;
      final stub = JSObject();
      stub.setProperty('Tensor'.toJS, realRoot.getProperty('Tensor'.toJS));
      stub.setProperty(
        'loadAndCompile'.toJS,
        ((JSAny bytes, JSObject options) {
          final accelerator =
              (options.getProperty('accelerator'.toJS) as JSString).toDart;
          if (accelerator == 'webgpu') {
            final promiseCtor = globalContext.getProperty<JSFunction>(
              'Promise'.toJS,
            );
            return promiseCtor.callAsConstructor<JSPromise<JSAny?>>(
              ((JSFunction resolve, JSFunction reject) {
                gpuResolve = resolve;
              }).toJS,
            );
          }
          return realRoot.callMethod<JSPromise<JSAny?>>(
            'loadAndCompile'.toJS,
            bytes,
            options,
          );
        }).toJS,
      );
      globalContext.setProperty('LiteRt'.toJS, stub);
      return (JSObject model) => gpuResolve!.callAsFunction(null, model);
    }

    test('a {gpu, cpu} request abandons a hung WebGPU compile and lands '
        'on WASM', () async {
      installHangingGpuRuntime();
      final cm = await cm_web.CompiledModel.fromBufferAsync(
        modelBytes,
        accelerators: const {Accelerator.gpu, Accelerator.cpu},
      ).timeout(const Duration(seconds: 20));
      expect(cm.accelerators, {Accelerator.cpu});
      final len = cm.inputByteSizes[0] ~/ 4;
      final input = Float32List(len)..fillRange(0, len, 1.0);
      expect((await cm.runAsync([input]))[0][0], closeTo(3.0, 1e-6));
      cm.close();
    });

    test('fromBufferWithGpuFallbackAsync surfaces the abandonment via '
        'onFallback', () async {
      installHangingGpuRuntime();
      Object? fallbackError;
      final cm = await cm_web.CompiledModel.fromBufferWithGpuFallbackAsync(
        modelBytes,
        onFallback: (e) => fallbackError = e,
      ).timeout(const Duration(seconds: 20));
      expect(cm.accelerators, {Accelerator.cpu});
      expect(fallbackError, isA<TimeoutException>());
      cm.close();
    });

    test('strict {gpu} requests are never timed out', () async {
      installHangingGpuRuntime();
      var settled = false;
      final pending = cm_web.CompiledModel.fromBufferAsync(
        modelBytes,
        accelerators: const {Accelerator.gpu},
      );
      unawaited(
        pending.then(
          (m) {
            settled = true;
            m.close();
          },
          onError: (Object _) {
            settled = true;
          },
        ),
      );
      // Wait out four watchdog windows; the strict request must still be
      // pending, surfacing the runtime's behavior rather than a synthetic
      // timeout.
      await Future<void>.delayed(const Duration(milliseconds: 1600));
      expect(settled, isFalse);
    });

    test('a late-settling abandoned WebGPU compile is disposed', () async {
      final settleGpu = installHangingGpuRuntime();
      final cm = await cm_web.CompiledModel.fromBufferAsync(
        modelBytes,
        accelerators: const {Accelerator.gpu, Accelerator.cpu},
      ).timeout(const Duration(seconds: 20));
      expect(cm.accelerators, {Accelerator.cpu});
      cm.close();

      var deleted = false;
      final fake = JSObject();
      fake.setProperty(
        'delete'.toJS,
        (() {
          deleted = true;
        }).toJS,
      );
      settleGpu(fake);
      final deadline = DateTime.now().add(const Duration(seconds: 5));
      while (!deleted && DateTime.now().isBefore(deadline)) {
        await Future<void>.delayed(const Duration(milliseconds: 50));
      }
      expect(
        deleted,
        isTrue,
        reason: 'the abandoned WebGPU model must be deleted when it settles',
      );
    });
  });
}
