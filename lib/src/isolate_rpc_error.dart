/// Thrown by an isolate RPC handler to send an EXACT error string to the
/// client, bypassing [serveIsolateRpc]'s default `'$e\n$st'` stringification.
///
/// Use this to preserve a wire-format error contract; for example, the generic
/// `Bad state: ...\n<stack>` wrapping would break a prefix the main side
/// string-matches via `startsWith`.
///
/// Lives in its own `dart:isolate`-free library so web/WASM consumers (e.g.
/// `throwDecodeFailure`) can reference it without pulling in the isolate
/// server implementation.
class IsolateRpcExactError implements Exception {
  /// The verbatim string sent as `{'id': id, 'error': message}`.
  final String message;
  const IsolateRpcExactError(this.message);
  @override
  String toString() => message;
}
