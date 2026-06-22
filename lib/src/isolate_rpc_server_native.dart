import 'dart:async';
import 'dart:isolate';

import 'isolate_rpc_error.dart';

/// Handler for a single RPC operation running inside the isolate.
///
/// Receives the raw request [message] map and returns the value to send back as
/// the response `result`, or throws to send an `error`.
typedef IsolateRpcHandler =
    Future<Object?> Function(Map<dynamic, dynamic> message);

/// Serves the standard `{id, op, ...}` request / `{id, result|error}` response
/// RPC protocol on [receivePort] — the isolate-side counterpart to
/// `IsolateRpcClient`.
///
/// For each incoming message it extracts `id` and `op`, dispatches to the
/// matching entry in [handlers], and replies `{id, result}` with the handler's
/// return value — or `{id, error}` with the stringified error and stack if the
/// handler throws. Messages that are not maps, or that lack `id`/`op`, are
/// ignored (matching the existing detector isolates).
///
/// [disposeOp] (default `'dispose'`) is handled specially: [onDispose] is
/// awaited (to free native resources), an ack `{id, result: null}` is sent so
/// the main side's `IsolateWorkerBase.disposeGracefully` can proceed, and
/// [receivePort] is closed. An unknown op replies with an error.
///
/// Call this from an `@pragma('vm:entry-point')` isolate entry after the initial
/// handshake (`mainSendPort.send(receivePort.sendPort)`):
///
/// ```dart
/// serveIsolateRpc(
///   mainSendPort: mainSendPort,
///   receivePort: receivePort,
///   handlers: {
///     'detect': (msg) async => (await core.detect(msg)).toMap(),
///   },
///   onDispose: () async => core.dispose(),
/// );
/// ```
void serveIsolateRpc({
  required SendPort mainSendPort,
  required ReceivePort receivePort,
  required Map<String, IsolateRpcHandler> handlers,
  String disposeOp = 'dispose',
  Future<void> Function()? onDispose,
}) {
  receivePort.listen((message) async {
    if (message is! Map) return;
    final int? id = message['id'] as int?;
    final String? op = message['op'] as String?;
    if (id == null || op == null) return;

    try {
      if (op == disposeOp) {
        if (onDispose != null) await onDispose();
        // Ack so the main side's disposeGracefully() can await teardown before
        // force-killing the isolate.
        mainSendPort.send({'id': id, 'result': null});
        receivePort.close();
        return;
      }

      final IsolateRpcHandler? handler = handlers[op];
      if (handler == null) {
        mainSendPort.send({'id': id, 'error': 'Unknown isolate op: $op'});
        return;
      }

      final Object? result = await handler(message);
      mainSendPort.send({'id': id, 'result': result});
    } catch (e, st) {
      final String error = e is IsolateRpcExactError ? e.message : '$e\n$st';
      mainSendPort.send({'id': id, 'error': error});
    }
  });
}
