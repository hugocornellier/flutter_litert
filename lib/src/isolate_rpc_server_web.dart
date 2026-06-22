import 'dart:async';

/// Handler for a single RPC operation running inside the isolate.
///
/// Receives the raw request [message] map and returns the value to send back as
/// the response `result`, or throws to send an `error`.
typedef IsolateRpcHandler =
    Future<Object?> Function(Map<dynamic, dynamic> message);

/// Web stub: isolate RPC serving requires `dart:isolate`, which is unavailable
/// on web/WASM (detectors run on the main isolate). Present only so the symbol
/// resolves on the web build; calling it throws.
///
/// The port parameters are typed [Object] (rather than `SendPort`/`ReceivePort`)
/// precisely because those types live in `dart:isolate`; only the native build
/// of this library is ever compiled with real ports.
void serveIsolateRpc({
  required Object mainSendPort,
  required Object receivePort,
  required Map<String, IsolateRpcHandler> handlers,
  String disposeOp = 'dispose',
  Future<void> Function()? onDispose,
}) {
  throw UnsupportedError(
    'serveIsolateRpc is native-only; dart:isolate is unavailable on web.',
  );
}
