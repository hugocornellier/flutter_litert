// Isolate-side RPC server ({id, op, ...} request / {id, result|error}
// response), the counterpart to IsolateRpcClient.
//
// Conditionally routed so the public API stays WASM-compatible: the
// serveIsolateRpc implementation lives in a dart:isolate-backed native library,
// with a throwing web stub. IsolateRpcExactError is re-exported from its own
// isolate-free library so web/WASM consumers can use it.
//
// The throwing web stub is the default (first) URI; the dart:isolate-backed
// native library is gated on dart.library.io. A static analyzer that does not
// evaluate the condition falls back to the first URI, so it must be the
// WASM-safe one.
export 'isolate_rpc_error.dart';
export 'isolate_rpc_server_web.dart'
    if (dart.library.io) 'isolate_rpc_server_native.dart';
