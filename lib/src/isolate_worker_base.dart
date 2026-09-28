import 'dart:isolate';

import 'isolate_rpc_client.dart';

/// Base class for isolate-backed worker objects.
///
/// Provides the standard boilerplate shared across single-channel isolate workers:
/// - An [IsolateRpcClient] for send/receive communication
/// - An [isReady] flag
/// - A [sendRequest] guard that rejects calls before initialization
/// - A [dispose] implementation that kills the isolate and closes the port
///
/// ## Lifecycle safety
///
/// [dispose] calls [IsolateRpcClient.failAllAndDispose] synchronously
/// (which internally calls `isolate.kill()`) before any `await`. This ensures
/// the background isolate is terminated immediately even if the caller does not
/// await the returned future, preventing use-after-free on native TFLite memory.
///
/// Subclasses must:
/// 1. Call `await initWorker(spawnFn)` during their own initialization to spawn
///    the isolate and complete the handshake.
/// 2. Override [workerDisposeOp] to specify the dispose operation name (if any).
/// 3. Override `dispose()` and call `await super.dispose()` when cleaning up.
abstract class IsolateWorkerBase {
  IsolateWorkerBase();

  final IsolateRpcClient rpc = IsolateRpcClient();
  bool _initialized = false;

  /// Returns true if the worker has been initialized and is ready for requests.
  bool get isReady => _initialized;

  /// Spawns the background isolate and completes the handshake.
  ///
  /// [spawnFn] receives the [SendPort] to pass to the isolate entry point and
  /// must return the spawned [Isolate]. The entry point must send its own
  /// [ReceivePort]'s [SendPort] back as the first message.
  ///
  /// On failure, the isolate is killed and the receive port closed before
  /// rethrowing.
  ///
  /// When [markReady] is true (the default), sets [isReady] to true after the
  /// handshake. Pass `false` for two-phase initialization where a subsequent
  /// operation must succeed before the worker is ready (use [markInitialized]
  /// to set [isReady] when appropriate).
  Future<void> initWorker(
    Future<Isolate> Function(SendPort sendPort) spawnFn, {
    Duration timeout = const Duration(seconds: 30),
    String timeoutMessage = 'Worker initialization timed out',
    void Function(dynamic)? onResponse,
    bool markReady = true,
  }) async {
    try {
      rpc.isolate = await spawnFn(rpc.receivePort.sendPort);

      rpc.sendPort = await setupIsolateHandshake(
        receivePort: rpc.receivePort,
        onResponse: onResponse ?? (msg) => rpc.handleResponse(msg),
        timeout: timeout,
        timeoutMessage: timeoutMessage,
      );

      if (markReady) _initialized = true;
    } catch (e) {
      rpc.isolate?.kill(priority: Isolate.immediate);
      rpc.receivePort.close();
      rethrow;
    }
  }

  /// Marks the worker as initialized and ready for requests.
  ///
  /// Subclasses that pass `markReady: false` to [initWorker] must call this
  /// once their additional initialization steps complete.
  void markInitialized() => _initialized = true;

  /// Sends a typed request to the background isolate.
  ///
  /// Throws [StateError] if the worker is not yet initialized.
  Future<T> sendRequest<T>(String operation, Map<String, dynamic> params) {
    if (!_initialized) {
      throw StateError('$runtimeType not initialized.');
    }
    return rpc.sendRequest<T>(operation, params);
  }

  /// Sends a typed request directly to the background isolate, bypassing the
  /// [isReady] guard.
  ///
  /// Use this only during initialization for operations that must succeed before
  /// the worker is marked ready (e.g., a two-phase 'init' request).
  Future<T> sendRequestUnchecked<T>(
    String operation,
    Map<String, dynamic> params,
  ) => rpc.sendRequest<T>(operation, params);

  /// The name of the operation to send to the isolate when disposing.
  ///
  /// Override in subclasses to specify the dispose operation (e.g., `'dispose'`).
  /// Return `null` if the isolate does not handle a dispose message.
  String? get workerDisposeOp => null;

  /// Disposes the worker: fails pending requests, sends [workerDisposeOp] to
  /// the isolate (if non-null), kills the isolate, and closes ports.
  ///
  /// Safe to call without awaiting, the isolate is killed synchronously.
  Future<void> dispose() async {
    rpc.failAllAndDispose(disposeOp: workerDisposeOp);
    _initialized = false;
  }

  /// Gracefully disposes the worker: sends [workerDisposeOp] to the isolate as a
  /// request and awaits its acknowledgement, giving the isolate a chance to
  /// free native resources before [dispose] force-kills it.
  ///
  /// [dispose] alone kills the isolate synchronously with
  /// `Isolate.kill(priority: immediate)`, which races past any queued dispose
  /// message, so the isolate can die before releasing its native TFLite
  /// interpreters (~10-26MB leaked per detector on Android; under sequential
  /// create/dispose load the low-memory killer reaps the process). Routing
  /// teardown through this method first lets the isolate ack and clean up.
  ///
  /// Best-effort: if the worker is not ready, [workerDisposeOp] is null, or the
  /// ack does not arrive within [timeout], it falls through to [dispose]. The
  /// isolate's dispose handler must reply `{'id': id, 'result': ...}` for the
  /// ack to resolve.
  Future<void> disposeGracefully({
    Duration timeout = const Duration(seconds: 5),
  }) async {
    final String? op = workerDisposeOp;
    if (_initialized && op != null) {
      try {
        await sendRequestUnchecked<dynamic>(
          op,
          const <String, dynamic>{},
        ).timeout(timeout);
      } catch (_) {
        // Fall through to the synchronous force-kill below.
      }
    }
    await dispose();
  }
}
