import 'dart:isolate';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Isolate entry that serves a few ops via [serveIsolateRpc].
@pragma('vm:entry-point')
void _serverEntry(SendPort main) {
  final ReceivePort rp = ReceivePort();
  main.send(rp.sendPort);
  serveIsolateRpc(
    mainSendPort: main,
    receivePort: rp,
    handlers: {
      'echo': (msg) async => 'echo:${msg['value']}',
      'add': (msg) async => (msg['a'] as int) + (msg['b'] as int),
      'boom': (msg) async => throw StateError('boom'),
      'exact': (msg) async =>
          throw IsolateRpcExactError('exact_prefix: ${msg['detail']}'),
    },
    onDispose: () async {},
  );
}

class _Worker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> start() => initWorker((sp) => Isolate.spawn(_serverEntry, sp));
}

void main() {
  test('serveIsolateRpc routes ops and returns results', () async {
    final w = _Worker();
    await w.start();
    expect(await w.sendRequest<String>('echo', {'value': 'hi'}), 'echo:hi');
    expect(await w.sendRequest<int>('add', {'a': 2, 'b': 3}), 5);
    await w.disposeGracefully();
  });

  test('serveIsolateRpc reports handler errors as error responses', () async {
    final w = _Worker();
    await w.start();
    await expectLater(
      w.sendRequest<dynamic>('boom', const {}),
      throwsA(isA<StateError>()),
    );
    await w.disposeGracefully();
  });

  test(
    'serveIsolateRpc sends IsolateRpcExactError verbatim (no wrapping)',
    () async {
      final w = _Worker();
      await w.start();
      // The wire error must be exactly the prefix string — no "Bad state:" and no
      // stack — so a main-side startsWith() contract survives.
      await expectLater(
        w.sendRequest<dynamic>('exact', const {'detail': 'hi'}),
        throwsA(
          predicate((e) => e is StateError && e.message == 'exact_prefix: hi'),
        ),
      );
      await w.disposeGracefully();
    },
  );

  test('serveIsolateRpc rejects unknown ops', () async {
    final w = _Worker();
    await w.start();
    await expectLater(
      w.sendRequest<dynamic>('nope', const {}),
      throwsA(isA<StateError>()),
    );
    await w.disposeGracefully();
  });

  test('serveIsolateRpc dispose acks so disposeGracefully resolves', () async {
    final w = _Worker();
    await w.start();
    await w.disposeGracefully().timeout(const Duration(seconds: 5));
    expect(w.isReady, isFalse);
  });
}
