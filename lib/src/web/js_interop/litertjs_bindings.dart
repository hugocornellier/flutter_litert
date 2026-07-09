// JS interop bindings for Google's LiteRT.js runtime
// (@litertjs/core v2.x, https://github.com/google-ai-edge/LiteRT/tree/main/litert/js).
//
// By default, [LiteRtInterpreter.fromBytes] auto-injects a loader script
// the first time it is called, so consumers don't have to add anything to
// their `web/index.html`. Override the URLs (e.g. for self-hosting / strict
// CSP) via [configureLiteRtWebLoader].
//
// Only the surface needed by [LiteRtInterpreter] is bound here.

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

/// Default `@litertjs/core` ESM bundle URL on jsDelivr.
const String _defaultLiteRtModuleUrl =
    'https://cdn.jsdelivr.net/npm/@litertjs/core@2.4.0/+esm';

/// Default WASM location passed to `LiteRt.loadLiteRt(...)`. A DIRECTORY
/// (trailing slash) so LiteRT.js's runtime feature probe picks the right
/// build per engine: the fast relaxed-SIMD `litert_wasm_internal` build on
/// Chrome/Firefox, and `litert_wasm_compat_internal` on Safari, which ships
/// relaxed SIMD behind a default-off flag and rejects the fast build at
/// parse time ("relaxed simd instructions not supported"). Pinning a full
/// `.js` file here would bypass that selection and break Safari entirely.
/// Threaded builds are only chosen when explicitly requested; they need
/// SharedArrayBuffer and thus COOP/COEP headers most hosts do not set.
const String _defaultLiteRtWasmUrl =
    'https://cdn.jsdelivr.net/npm/@litertjs/core@2.4.0/wasm/';

String _liteRtModuleUrl = _defaultLiteRtModuleUrl;
String _liteRtWasmUrl = _defaultLiteRtWasmUrl;
bool _autoLoaderEnabled = true;
Future<void>? _injectedLoadFuture;

/// Configure the URLs / behavior of the LiteRT.js web auto-loader.
///
/// Deprecated in favor of [configureLiteRtWebLoader], the public web-specific
/// spelling.
@Deprecated(
  'Use configureLiteRtWebLoader instead. '
  'This alias will be removed in a future major release.',
)
void configureLiteRtLoader({
  String? moduleUrl,
  String? wasmUrl,
  bool? autoLoad,
}) {
  configureLiteRtWebLoader(
    moduleUrl: moduleUrl,
    wasmUrl: wasmUrl,
    autoLoad: autoLoad,
  );
}

/// Configure the LiteRT.js web auto-loader.
///
/// Call it once before the first `LiteRtInterpreter.fromBytes(...)` when you
/// need to self-host the LiteRT.js module/WASM files or disable auto-loading.
///
/// - [moduleUrl]: ESM URL for the `@litertjs/core` bundle.
/// - [wasmUrl]: URL passed to `LiteRt.loadLiteRt(...)`. Either a directory
///   ending in `/` (lets LiteRT.js auto-detect SIMD/threaded) or a specific
///   `.js` file.
/// - [autoLoad]: when `false`, disables auto-injection. The host page is
///   then responsible for loading the runtime and assigning it to
///   `window.LiteRt`.
void configureLiteRtWebLoader({
  String? moduleUrl,
  String? wasmUrl,
  bool? autoLoad,
}) {
  if (moduleUrl != null) _liteRtModuleUrl = moduleUrl;
  if (wasmUrl != null) _liteRtWasmUrl = wasmUrl;
  if (autoLoad != null) _autoLoaderEnabled = autoLoad;
}

@JS('window.LiteRt')
external JSObject? get _liteRtRoot;

@JS('window.LiteRtReady')
external JSAny? get _liteRtReady;

@JS('window.LiteRtLoadError')
external JSAny? get _liteRtLoadError;

/// Returns true once the page-level LiteRT.js loader has finished initializing.
bool isLiteRtReady() {
  final ready = _liteRtReady;
  if (ready == null || !ready.isA<JSBoolean>()) return false;
  return (ready as JSBoolean).toDart;
}

/// Page-level load error (window.LiteRtLoadError), or null if none.
String? liteRtLoadError() {
  final err = _liteRtLoadError;
  if (err == null) return null;
  if (err.isA<JSString>()) return (err as JSString).toDart;
  return err.dartify()?.toString();
}

/// Awaits the host page's `litert-ready` event, or auto-injects a loader
/// script if no host-page loader has been wired and auto-loading is enabled.
///
/// Three resolved states are possible:
///  - window.LiteRtReady === true: resolves immediately
///  - window.LiteRtReady === false: the loader ran and failed; throws now
///  - window.LiteRtReady undefined: no loader yet; auto-inject (if
///    enabled), then wait for the event
Future<void> waitForLiteRt({Duration timeout = const Duration(seconds: 30)}) {
  final ready = _liteRtReady;
  if (ready != null && ready.isA<JSBoolean>()) {
    if ((ready as JSBoolean).toDart) {
      return Future<void>.value();
    }
    final err = liteRtLoadError() ?? 'unknown error';
    return Future<void>.error(
      StateError('LiteRT.js failed to initialize: $err'),
    );
  }

  // No host-page loader detected. Auto-inject one if enabled.
  if (_autoLoaderEnabled) {
    _injectedLoadFuture ??= _injectLoaderScript();
  }

  // Race the `litert-ready` event against the timeout.
  final completer = Completer<void>();
  void handler(JSAny _) {
    if (!completer.isCompleted) completer.complete();
  }

  globalContext.callMethod<JSAny?>(
    'addEventListener'.toJS,
    'litert-ready'.toJS,
    handler.toJS,
  );
  return completer.future
      .timeout(
        timeout,
        onTimeout: () => throw StateError(
          'LiteRT.js did not finish loading within ${timeout.inSeconds}s. '
          'Check network access to cdn.jsdelivr.net (or call '
          'configureLiteRtWebLoader to point at a self-hosted bundle).',
        ),
      )
      .then((_) {
        if (!isLiteRtReady()) {
          final err = liteRtLoadError() ?? 'unknown error';
          throw StateError('LiteRT.js failed to initialize: $err');
        }
      });
}

/// Programmatically appends a `<script type="module">` to `<head>` that
/// dynamically `import()`s `@litertjs/core`, calls `loadLiteRt(...)`, and
/// dispatches the `litert-ready` event the same way the documented manual
/// loader snippet does. Idempotent across the page lifetime via
/// [_injectedLoadFuture].
Future<void> _injectLoaderScript() {
  final completer = Completer<void>();
  final doc = globalContext['document'] as JSObject;
  final head = doc.getProperty<JSObject?>('head'.toJS);
  if (head == null) {
    completer.completeError(
      StateError(
        'document.head is not available; cannot auto-inject the LiteRT.js '
        'loader. Provide a host-page <script> instead.',
      ),
    );
    return completer.future;
  }

  // The body runs as a module; bare imports go through jsdelivr's `+esm`
  // bundle so they resolve in the browser without an import map.
  final String body =
      "const moduleUrl = ${_jsString(_liteRtModuleUrl)};\n"
      "const wasmUrl = ${_jsString(_liteRtWasmUrl)};\n"
      "try {\n"
      "  const mod = await import(moduleUrl);\n"
      "  await mod.loadLiteRt(wasmUrl);\n"
      "  window.LiteRt = mod;\n"
      "  window.LiteRtReady = true;\n"
      "} catch (e) {\n"
      "  console.error('flutter_litert: LiteRT.js load failed:', e);\n"
      "  window.LiteRtReady = false;\n"
      "  window.LiteRtLoadError = e && e.message ? e.message : String(e);\n"
      "} finally {\n"
      "  window.dispatchEvent(new Event('litert-ready'));\n"
      "}";

  final script = doc.callMethod<JSObject>('createElement'.toJS, 'script'.toJS);
  script['type'] = 'module'.toJS;
  script['textContent'] = body.toJS;
  head.callMethod<JSAny?>('appendChild'.toJS, script);

  completer.complete();
  return completer.future;
}

/// Encodes a Dart string as a JS string literal (with double quotes).
String _jsString(String s) {
  final escaped = s
      .replaceAll(r'\', r'\\')
      .replaceAll('"', r'\"')
      .replaceAll('\n', r'\n')
      .replaceAll('\r', r'\r');
  return '"$escaped"';
}

/// Calls `LiteRt.loadAndCompile(bytes, {accelerator})`.
Future<CompiledModelJS> loadAndCompile(
  Uint8List bytes, {
  String accelerator = 'wasm',
}) async {
  final root = _liteRtRoot;
  if (root == null) {
    throw StateError(
      'LiteRt is not loaded on window. Did the page-level loader run?',
    );
  }
  final options = JSObject();
  options['accelerator'] = accelerator.toJS;
  final JSPromise<CompiledModelJS> promise = root
      .callMethod<JSPromise<CompiledModelJS>>(
        'loadAndCompile'.toJS,
        bytes.toJS,
        options,
      );
  return await promise.toDart;
}

/// Wrapper for a `LiteRt.Tensor`.
extension type LiteRtTensorJS._(JSObject _) implements JSObject {
  external void delete();

  /// Sync TypedArray view. Throws for GPU-resident tensors; use [dataAsync].
  external JSAny toTypedArray();

  /// Async TypedArray copy. Works for both CPU and GPU tensors.
  @JS('data')
  external JSPromise<JSAny> dataAsync();
}

/// Wrapper for `LiteRt.CompiledModel`.
extension type CompiledModelJS._(JSObject _) implements JSObject {
  external void delete();

  /// `run(input | input[])` returns `Promise<Tensor[]>`.
  external JSPromise<JSArray<LiteRtTensorJS>> run(JSAny input);

  external JSArray<JSObject> getInputDetails();
  external JSArray<JSObject> getOutputDetails();
}

/// Constructs a CPU/RAM-backed input tensor: `new LiteRt.Tensor(typedArray, shape)`.
LiteRtTensorJS makeInputTensor(Float32List data, List<int> shape) {
  final root = _liteRtRoot;
  if (root == null) {
    throw StateError('LiteRt is not loaded on window.');
  }
  final tensorCtor = root['Tensor'] as JSFunction;
  final shapeJs = (Int32List.fromList(shape)).toJS;
  return tensorCtor.callAsConstructor<LiteRtTensorJS>(data.toJS, shapeJs);
}
