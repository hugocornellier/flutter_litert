import 'dart:isolate';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Test isolate entry that completes the handshake and ACKs a 'dispose' op
/// (mirrors the real detector isolates: reply `{id, result}` then close).
@pragma('vm:entry-point')
void _ackingEntry(SendPort main) {
  final ReceivePort rp = ReceivePort();
  main.send(rp.sendPort);
  rp.listen((msg) {
    if (msg is! Map) return;
    final id = msg['id'];
    final op = msg['op'];
    if (op == 'dispose') {
      main.send({'id': id, 'result': null});
      rp.close();
    } else {
      main.send({'id': id, 'result': 'echo:$op'});
    }
  });
}

/// Test isolate entry that NEVER acks 'dispose' (closes silently). Exercises the
/// timeout-then-force-kill fallback path of [IsolateWorkerBase.disposeGracefully].
@pragma('vm:entry-point')
void _silentEntry(SendPort main) {
  final ReceivePort rp = ReceivePort();
  main.send(rp.sendPort);
  rp.listen((msg) {
    if (msg is! Map) return;
    if (msg['op'] == 'dispose') {
      rp.close(); // no ack
    }
  });
}

class _TestWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> start(void Function(SendPort) entry) =>
      initWorker((sp) => Isolate.spawn(entry, sp));
}

void main() {
  test('disposeGracefully resolves once the isolate acks dispose', () async {
    final w = _TestWorker();
    await w.start(_ackingEntry);
    expect(w.isReady, isTrue);

    await w.disposeGracefully().timeout(const Duration(seconds: 5));
    expect(w.isReady, isFalse);
  });

  test(
    'disposeGracefully falls through to force-kill when no ack arrives',
    () async {
      final w = _TestWorker();
      await w.start(_silentEntry);

      final sw = Stopwatch()..start();
      await w.disposeGracefully(timeout: const Duration(milliseconds: 200));
      sw.stop();

      expect(w.isReady, isFalse);
      // Must fall through promptly after the timeout, not hang for seconds.
      expect(sw.elapsedMilliseconds, lessThan(3000));
    },
  );

  test('disposeGracefully on a never-started worker just disposes', () async {
    final w = _TestWorker();
    expect(w.isReady, isFalse);
    await w.disposeGracefully(timeout: const Duration(milliseconds: 200));
    expect(w.isReady, isFalse);
  });
}
