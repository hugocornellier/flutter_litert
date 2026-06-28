// Wraps camera-frame bytes for an isolate request payload, choosing a
// platform-appropriate carrier.
//
// Routed conditionally so the public API stays WASM-compatible: the native
// build uses dart:isolate's TransferableTypedData for zero-copy transfer, while
// the web build (no dart:isolate) falls back to the raw bytes.
//
// The dart:isolate-free web carrier is the default (first) URI; the native
// carrier is gated on dart.library.io. A static analyzer that does not evaluate
// the condition falls back to the first URI, so it must be the WASM-safe one.
export 'transferable_bytes_web.dart'
    if (dart.library.io) 'transferable_bytes_native.dart';
