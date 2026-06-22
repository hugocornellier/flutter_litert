// Isolate-side RPC server ({id, op, ...} request / {id, result|error}
// response), the counterpart to IsolateRpcClient.
//
// Conditionally routed so the public API stays WASM-compatible: the
// serveIsolateRpc implementation lives in a dart:isolate-backed native library,
// with a throwing web stub. IsolateRpcExactError is re-exported from its own
// isolate-free library so web/WASM consumers can use it.
export 'isolate_rpc_error.dart';
export 'isolate_rpc_server_native.dart'
    if (dart.library.js_interop) 'isolate_rpc_server_web.dart';
