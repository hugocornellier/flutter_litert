// Wraps camera-frame bytes for an isolate request payload, choosing a
// platform-appropriate carrier.
//
// Routed conditionally so the public API stays WASM-compatible: the native
// build uses dart:isolate's TransferableTypedData for zero-copy transfer, while
// the web build (no dart:isolate) falls back to the raw bytes.
export 'transferable_bytes_native.dart'
    if (dart.library.js_interop) 'transferable_bytes_web.dart';
