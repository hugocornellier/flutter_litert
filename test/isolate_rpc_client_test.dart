import 'dart:isolate';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Isolate that serves via [serveIsolateRpc] (which acks dispose).
@pragma('vm:entry-point')
void _ackEntry(SendPort main) {
  final ReceivePort rp = ReceivePort();
  main.send(rp.sendPort);
  serveIsolateRpc(
    mainSendPort: main,
    receivePort: rp,
    handlers: {'echo': (m) async => 'echo:${m['value']}'},
    onDispose: () async {},
  );
}

Future<IsolateRpcClient> _spawn() async {
  final rpc = IsolateRpcClient();
  rpc.isolate = await Isolate.spawn(_ackEntry, rpc.receivePort.sendPort);
  rpc.sendPort = await setupIsolateHandshake(
    receivePort: rpc.receivePort,
    onResponse: rpc.handleResponse,
    timeout: const Duration(seconds: 5),
    timeoutMessage: 'init timeout',
  );
  return rpc;
}

void main() {
  test(
    'disposeGracefully sends dispose, awaits the ack, then force-disposes',
    () async {
      final rpc = await _spawn();
      expect(await rpc.sendRequest<String>('echo', {'value': 'x'}), 'echo:x');
      await rpc
          .disposeGracefully(disposeOp: 'dispose')
          .timeout(const Duration(seconds: 5));
      expect(rpc.sendPort, isNull); // failAllAndDispose ran
    },
  );

  test('disposeGracefully with a null op just force-disposes', () async {
    final rpc = await _spawn();
    await rpc.disposeGracefully().timeout(const Duration(seconds: 5));
    expect(rpc.sendPort, isNull);
  });
}
