import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:web/web.dart' as web;

import 'litertjs_interpreter.dart' show LiteRtRuntimeError;

/// Decodes encoded image bytes (JPEG, PNG, etc.) to an [web.ImageBitmap].
///
/// Uses `createImageBitmap`, which decodes off the main thread and avoids the
/// HTMLImageElement load-event roundtrip. Returns null if decoding fails.
Future<web.ImageBitmap?> decodeBitmap(Uint8List bytes) async {
  final web.Blob blob = web.Blob([bytes.toJS].toJS);
  try {
    return await web.window.createImageBitmap(blob).toDart;
  } catch (_) {
    return null;
  }
}

/// Resolves a requested LiteRT.js accelerator (`'auto'` / `'webgpu'` /
/// `'wasm'`) into a concrete backend for this browser.
///
/// Explicit values pass through untouched. `'auto'` resolves to `'webgpu'`
/// only when the browser is Chromium-based AND `navigator.gpu` yields a
/// hardware adapter; everything else resolves to `'wasm'`.
///
/// The Chromium gate is deliberate: LiteRT.js's WebGPU delegate is developed
/// and tuned against Chrome's Dawn. Firefox 152 (and Safari) expose a WebGPU
/// that compiles and runs small vision models without ever throwing, but at
/// unusable speed (measured 22x slower than single-threaded WASM SIMD on
/// Firefox 152 / Apple Silicon), so API presence alone must not select it.
/// Callers can still force `'webgpu'` explicitly, and
/// [WebGpuFallback.maybeSwapIfWebGpuSlow] catches slow-but-functional stacks
/// that slip through.
///
/// The `'auto'` probe runs once per page load and is cached.
Future<String> resolveWebAccelerator(String requested) {
  if (requested != 'auto') return Future<String>.value(requested);
  return _cachedAutoDecision ??= _probeAuto();
}

Future<String>? _cachedAutoDecision;

/// Test hook: clears the cached `'auto'` decision.
void debugResetAcceleratorResolution() {
  _cachedAutoDecision = null;
}

Future<String> _probeAuto() async {
  final String decision = await _probeAutoInner();
  debugPrint("flutter_litert: 'auto' accelerator resolved to '$decision'");
  return decision;
}

Future<String> _probeAutoInner() async {
  try {
    final JSObject? nav = globalContext['navigator'] as JSObject?;
    if (nav == null) return 'wasm';

    // Chromium gate. `navigator.userAgentData` only exists on Chromium, and
    // the UA-string check backstops configurations that disable it.
    final String ua = ((nav['userAgent'] as JSString?)?.toDart) ?? '';
    final bool isChromium = nav.has('userAgentData') || ua.contains('Chrome/');
    if (!isChromium) return 'wasm';

    final JSAny? gpu = nav['gpu'];
    if (gpu == null || !gpu.isA<JSObject>()) return 'wasm';

    final JSAny? adapter = await (gpu as JSObject)
        .callMethod<JSPromise<JSAny?>>('requestAdapter'.toJS)
        .toDart
        .timeout(const Duration(seconds: 3), onTimeout: () => null);
    if (adapter == null || !adapter.isA<JSObject>()) return 'wasm';

    // Reject software adapters (headless CI, GPU-less machines): WASM SIMD
    // beats a software-rasterized "GPU" for these models.
    final JSAny? info = (adapter as JSObject)['info'];
    if (info != null && info.isA<JSObject>()) {
      final JSObject i = info as JSObject;
      final String vendor = ((i['vendor'] as JSString?)?.toDart ?? '')
          .toLowerCase();
      final String arch = ((i['architecture'] as JSString?)?.toDart ?? '')
          .toLowerCase();
      if (vendor.contains('swiftshader') ||
          arch.contains('swiftshader') ||
          arch.contains('llvmpipe')) {
        return 'wasm';
      }
    }
    return 'webgpu';
  } catch (_) {
    return 'wasm';
  }
}

/// Logs when LiteRT.js silently compiled a model on a different backend than
/// requested (its compile-time WebGPU-to-WASM fallback), so the swap is
/// visible in the console instead of only in perf traces.
void logCompileFallback({
  required String model,
  required String requested,
  required String actual,
}) {
  if (requested == actual) return;
  debugPrint(
    'flutter_litert: $model requested $requested but LiteRT.js compiled it '
    'on $actual (compile-time fallback).',
  );
}

/// Mixin that adds transparent WebGPU-to-WASM runtime fallback to a web
/// detector class.
///
/// Apply with `with WebGpuFallback`. The applying class must provide:
/// - `String? get activeAccelerator`: the current backend
/// - `Future<void> swapToWasm()`: dispose and re-init all runners on WASM
///
/// Then wrap each public inference call with [withFallback]:
/// ```dart
/// Future<List<Result>> detect(Uint8List bytes) async {
///   ...
///   return withFallback(() => _detectInner(bytes));
/// }
/// ```
///
/// For `'auto'` initializations that landed on WebGPU, also call
/// [maybeSwapIfWebGpuSlow] once after init with a representative probe
/// inference: it catches engines whose WebGPU works but is unusably slow,
/// which the error-driven path can never see.
mixin WebGpuFallback {
  bool _fellBackToWasm = false;

  /// True once the detector has irreversibly fallen back from WebGPU to WASM
  /// (after a runtime GPU error, or a failed warmup budget).
  bool get fellBackToWasm => _fellBackToWasm;

  /// The accelerator currently in use. Provided by the applying class.
  String? get activeAccelerator;

  /// Disposes and re-initializes all model runners on WASM. Called once on
  /// the first runtime GPU error. Provided by the applying class.
  Future<void> swapToWasm();

  /// Runs [fn]. If a LiteRT runtime error occurs on the WebGPU path,
  /// transparently swaps all runners to WASM via [swapToWasm] and retries
  /// [fn] once. Non-LiteRT errors (logic bugs, bad input) are rethrown
  /// untouched so they surface instead of masquerading as backend fallbacks.
  Future<T> withFallback<T>(Future<T> Function() fn) async {
    try {
      return await fn();
    } on LiteRtRuntimeError catch (e) {
      if (activeAccelerator == 'webgpu' && !_fellBackToWasm) {
        debugPrint(
          'flutter_litert: runtime GPU error on WebGPU; swapping all runners '
          'to WASM. Cause: $e',
        );
        await swapToWasm();
        _fellBackToWasm = true;
        return fn();
      }
      rethrow;
    }
  }

  /// Times [probe] (one representative inference) and swaps every runner to
  /// WASM when the median of [timedRuns] exceeds [budgetMs].
  ///
  /// Call once after an `'auto'` initialization that landed on WebGPU. The
  /// default 50ms budget sits far above healthy WebGPU latencies for small
  /// vision models (2-5ms on Chrome's Dawn) and far below pathological ones
  /// (~200ms per inference on Firefox 152), so it separates the two without
  /// tuning. No-op when the current backend is not WebGPU.
  Future<void> maybeSwapIfWebGpuSlow({
    required Future<void> Function() probe,
    double budgetMs = 50.0,
    int warmupRuns = 2,
    int timedRuns = 3,
  }) async {
    if (activeAccelerator != 'webgpu') return;
    try {
      for (int i = 0; i < warmupRuns; i++) {
        await probe();
      }
      final List<double> timesMs = <double>[];
      for (int i = 0; i < timedRuns; i++) {
        final Stopwatch sw = Stopwatch()..start();
        await probe();
        sw.stop();
        timesMs.add(sw.elapsedMicroseconds / 1000.0);
      }
      timesMs.sort();
      final double medianMs = timesMs[timesMs.length ~/ 2];
      if (medianMs > budgetMs) {
        debugPrint(
          'flutter_litert: WebGPU warmup median ${medianMs.toStringAsFixed(1)}'
          'ms exceeds the ${budgetMs.toStringAsFixed(0)}ms budget; switching '
          'to WASM.',
        );
        await swapToWasm();
        _fellBackToWasm = true;
      } else {
        debugPrint(
          'flutter_litert: WebGPU warmup median '
          '${medianMs.toStringAsFixed(1)}ms; keeping WebGPU.',
        );
      }
    } on LiteRtRuntimeError catch (e) {
      // A GPU failure this early is the same signal, just louder.
      debugPrint(
        'flutter_litert: GPU error during WebGPU warmup; switching to WASM. '
        'Cause: $e',
      );
      await swapToWasm();
      _fellBackToWasm = true;
    }
  }
}
