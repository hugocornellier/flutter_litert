@TestOn('browser')
library;

import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Unit coverage for the web hardening helpers: the [WebGpuFallback] mixin's
/// runtime-error and warmup-budget fallbacks (driven by a fake detector, so
/// every branch runs deterministically with or without a GPU) and
/// [resolveWebAccelerator]'s 'auto' gating (driven by stubbing
/// window.navigator, which is a replaceable attribute per WebIDL).
class _FakeDetector with WebGpuFallback {
  _FakeDetector(this._accelerator);

  String? _accelerator;
  int swapCalls = 0;

  @override
  String? get activeAccelerator => _accelerator;

  @override
  Future<void> swapToWasm() async {
    swapCalls++;
    _accelerator = 'wasm';
  }
}

LiteRtRuntimeError _gpuError() =>
    LiteRtRuntimeError(accelerator: 'webgpu', cause: 'device lost (test)');

JSObject _makeAdapter({String vendor = 'apple', String architecture = ''}) {
  final adapter = JSObject();
  final info = JSObject();
  info['vendor'] = vendor.toJS;
  info['architecture'] = architecture.toJS;
  adapter['info'] = info;
  return adapter;
}

JSObject _makeGpu({JSObject? adapter}) {
  final gpu = JSObject();
  gpu['requestAdapter'] = (() => Future<JSAny?>.value(adapter).toJS).toJS;
  return gpu;
}

JSObject _makeNavigator({
  required String userAgent,
  bool chromium = false,
  JSObject? gpu,
}) {
  final nav = JSObject();
  nav['userAgent'] = userAgent.toJS;
  if (chromium) nav['userAgentData'] = JSObject();
  if (gpu != null) nav['gpu'] = gpu;
  return nav;
}

void main() {
  group('WebGpuFallback.withFallback', () {
    test('passes results through and never swaps on success', () async {
      final d = _FakeDetector('webgpu');
      expect(await d.withFallback(() async => 42), 42);
      expect(d.swapCalls, 0);
      expect(d.fellBackToWasm, isFalse);
    });

    test('swaps to WASM and retries once on a GPU runtime error', () async {
      final d = _FakeDetector('webgpu');
      var calls = 0;
      final result = await d.withFallback(() async {
        calls++;
        if (calls == 1) throw _gpuError();
        return 'ok';
      });
      expect(result, 'ok');
      expect(calls, 2);
      expect(d.swapCalls, 1);
      expect(d.fellBackToWasm, isTrue);
      expect(d.activeAccelerator, 'wasm');
    });

    test('propagates the retry failure without a second swap', () async {
      final d = _FakeDetector('webgpu');
      var calls = 0;
      await expectLater(
        d.withFallback(() async {
          calls++;
          throw _gpuError();
        }),
        throwsA(isA<LiteRtRuntimeError>()),
      );
      expect(calls, 2);
      expect(d.swapCalls, 1);
      expect(d.fellBackToWasm, isTrue);
    });

    test('rethrows GPU errors when already on WASM', () async {
      final d = _FakeDetector('wasm');
      await expectLater(
        d.withFallback(() async => throw _gpuError()),
        throwsA(isA<LiteRtRuntimeError>()),
      );
      expect(d.swapCalls, 0);
      expect(d.fellBackToWasm, isFalse);
    });

    test(
      'rethrows non-LiteRT errors untouched (no fallback masking)',
      () async {
        final d = _FakeDetector('webgpu');
        await expectLater(
          d.withFallback(() async => throw ArgumentError('bad input')),
          throwsArgumentError,
        );
        expect(d.swapCalls, 0);
        expect(d.fellBackToWasm, isFalse);
      },
    );
  });

  group('WebGpuFallback.maybeSwapIfWebGpuSlow', () {
    test('no-ops when the backend is not WebGPU', () async {
      final d = _FakeDetector('wasm');
      var probes = 0;
      await d.maybeSwapIfWebGpuSlow(probe: () async => probes++);
      expect(probes, 0);
      expect(d.swapCalls, 0);
    });

    test('keeps WebGPU when the median beats the budget', () async {
      final d = _FakeDetector('webgpu');
      var probes = 0;
      await d.maybeSwapIfWebGpuSlow(
        probe: () async => probes++,
        budgetMs: 10000,
      );
      expect(probes, 5); // 2 warmup + 3 timed by default.
      expect(d.swapCalls, 0);
      expect(d.fellBackToWasm, isFalse);
      expect(d.activeAccelerator, 'webgpu');
    });

    test('swaps to WASM when the median exceeds the budget', () async {
      final d = _FakeDetector('webgpu');
      await d.maybeSwapIfWebGpuSlow(
        probe: () => Future<void>.delayed(const Duration(milliseconds: 8)),
        budgetMs: 1,
      );
      expect(d.swapCalls, 1);
      expect(d.fellBackToWasm, isTrue);
      expect(d.activeAccelerator, 'wasm');
    });

    test('honors custom warmup and timed run counts', () async {
      final d = _FakeDetector('webgpu');
      var probes = 0;
      await d.maybeSwapIfWebGpuSlow(
        probe: () async => probes++,
        budgetMs: 10000,
        warmupRuns: 0,
        timedRuns: 1,
      );
      expect(probes, 1);
    });

    test('a GPU error during warmup swaps to WASM', () async {
      final d = _FakeDetector('webgpu');
      await d.maybeSwapIfWebGpuSlow(probe: () async => throw _gpuError());
      expect(d.swapCalls, 1);
      expect(d.fellBackToWasm, isTrue);
    });
  });

  group('resolveWebAccelerator', () {
    // window.navigator only has a prototype getter, so assignment is
    // refused; Object.defineProperty on the window instance shadows the
    // getter with an own property, and deleting that property restores the
    // real navigator.
    tearDown(() {
      globalContext.delete('navigator'.toJS);
      debugResetAcceleratorResolution();
    });

    void stubNavigator(JSObject nav) {
      final descriptor = JSObject();
      descriptor['value'] = nav;
      descriptor['configurable'] = true.toJS;
      (globalContext['Object'] as JSObject).callMethod(
        'defineProperty'.toJS,
        globalContext,
        'navigator'.toJS,
        descriptor,
      );
      final applied =
          ((globalContext['navigator'] as JSObject?)?['userAgent'] as JSString?)
              ?.toDart;
      expect(
        applied,
        (nav['userAgent'] as JSString).toDart,
        reason: 'the navigator stub must be visible through window',
      );
      debugResetAcceleratorResolution();
    }

    test('explicit values pass through untouched', () async {
      expect(await resolveWebAccelerator('wasm'), 'wasm');
      expect(await resolveWebAccelerator('webgpu'), 'webgpu');
    });

    test("'auto' resolves to a concrete backend and caches", () async {
      debugResetAcceleratorResolution();
      final first = await resolveWebAccelerator('auto');
      expect(first, anyOf('wasm', 'webgpu'));
      expect(await resolveWebAccelerator('auto'), first);
    });

    test("'auto' rejects non-Chromium browsers even with a GPU", () async {
      stubNavigator(
        _makeNavigator(
          userAgent: 'Mozilla/5.0 Gecko/20100101 Firefox/152.0',
          gpu: _makeGpu(adapter: _makeAdapter()),
        ),
      );
      expect(await resolveWebAccelerator('auto'), 'wasm');
    });

    test("'auto' falls back to WASM when navigator.gpu is missing", () async {
      stubNavigator(
        _makeNavigator(userAgent: 'Chrome/149.0 test', chromium: true),
      );
      expect(await resolveWebAccelerator('auto'), 'wasm');
    });

    test("'auto' falls back to WASM when no adapter is granted", () async {
      stubNavigator(
        _makeNavigator(
          userAgent: 'Chrome/149.0 test',
          chromium: true,
          gpu: _makeGpu(adapter: null),
        ),
      );
      expect(await resolveWebAccelerator('auto'), 'wasm');
    });

    test("'auto' rejects software (SwiftShader/llvmpipe) adapters", () async {
      stubNavigator(
        _makeNavigator(
          userAgent: 'Chrome/149.0 test',
          chromium: true,
          gpu: _makeGpu(adapter: _makeAdapter(vendor: 'Google SwiftShader')),
        ),
      );
      expect(await resolveWebAccelerator('auto'), 'wasm');
      stubNavigator(
        _makeNavigator(
          userAgent: 'Chrome/149.0 test',
          chromium: true,
          gpu: _makeGpu(adapter: _makeAdapter(architecture: 'llvmpipe')),
        ),
      );
      expect(await resolveWebAccelerator('auto'), 'wasm');
    });

    test("'auto' picks WebGPU on Chromium with a hardware adapter", () async {
      stubNavigator(
        _makeNavigator(
          userAgent: 'Chrome/149.0 test',
          chromium: true,
          gpu: _makeGpu(adapter: _makeAdapter(vendor: 'apple')),
        ),
      );
      expect(await resolveWebAccelerator('auto'), 'webgpu');
    });
  });
}
