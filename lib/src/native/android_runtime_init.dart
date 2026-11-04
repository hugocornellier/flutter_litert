import 'dart:async';
import 'dart:ffi';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../android_runtime.dart';
import '../bindings/android_runtime_shim.dart';

const MethodChannel _channel = MethodChannel('flutter_litert');

Future<LiteRtRuntimeStatus>? _pending;
bool? _pendingEnableGpu;
bool? _pendingEnableAutomaticDownload;
DateTime? _pendingDeadline;

/// How much longer than the plugin's own timeout to wait for its reply.
const Duration _replyGrace = Duration(seconds: 30);

/// The LiteRT runtime this app uses.
///
/// [LiteRtRuntime.playServices] on Android apps built with
/// `flutterLitert.runtime=playServices`, and with `auto` once
/// [initializeAndroidRuntime] chose Play services. Null in an `auto` build
/// until [initializeAndroidRuntime] has chosen. [LiteRtRuntime.bundled]
/// otherwise.
LiteRtRuntime? get androidRuntime {
  final shim = androidRuntimeShim;
  if (shim == null) return LiteRtRuntime.bundled;
  return switch (shim.selected) {
    kRuntimePlayServices => LiteRtRuntime.playServices,
    kRuntimeBundled => LiteRtRuntime.bundled,
    _ => null,
  };
}

/// Initializes the Google Play services LiteRT runtime.
///
/// Apps built with `flutterLitert.runtime=playServices` or `auto` must await
/// this, from the main isolate, before creating an `Interpreter` or a
/// `CompiledModel`. Isolates spawned afterwards can use both without calling it
/// again. With the bundled runtime, and on every platform but Android, it
/// returns [LiteRtRuntimeStatus.bundled] immediately.
///
/// In an `auto` build the first call chooses the runtime for the process: Play
/// services when both of its halves initialize within [timeout], and the
/// bundled runtime otherwise, with [LiteRtRuntimeStatus.playServicesError]
/// saying why. It never throws [LiteRtRuntimeUnavailableException]. Later
/// calls keep that choice, so a GPU request that Play services could not meet
/// does not switch to the bundled runtime. A short [timeout] lets the app
/// start on the bundled runtime while Play services is still downloading its
/// module.
///
/// [enableGpu] asks Play services for its GPU accelerator, for
/// `CompiledModel` with `Accelerator.gpu`. If the GPU request fails, the
/// CompiledModel runtime is initialized for CPU only and
/// [LiteRtRuntimeStatus.gpuError] says why. The first successful
/// initialization fixes this for the process: asking for the GPU later does
/// not add it, and the status of that later call says so. Play services may
/// also initialize without the GPU while it downloads its GPU module, in which
/// case a later launch of the app can have it.
///
/// [enableAutomaticDownload] lets Play services download the runtime module
/// when the device does not have it yet. Without it, initialization fails
/// until the module is installed some other way.
///
/// [timeout] bounds the call. A runtime half that Play services has not
/// initialized by then is reported as failed, typically because the module
/// download is slow or stalled, and the next call retries it.
///
/// Concurrent calls with the same options share one initialization; a call
/// with other options runs after it. Each call stays within its own [timeout].
/// A runtime half that failed is retried by the next call. Throws
/// [LiteRtRuntimeUnavailableException] when neither half could be initialized;
/// when only one could, the returned status says which and why.
Future<LiteRtRuntimeStatus> initializeAndroidRuntime({
  bool enableGpu = false,
  bool enableAutomaticDownload = true,
  Duration timeout = const Duration(minutes: 5),
}) {
  final shim = androidRuntimeShim;
  if (shim == null || shim.mode == kRuntimeBundled) {
    return Future.value(const LiteRtRuntimeStatus.bundled());
  }
  if (shim.isAuto) {
    switch (shim.selected) {
      case kRuntimeBundled:
        return Future.value(
          LiteRtRuntimeStatus.bundled(
            playServicesError: shim.playServicesError,
          ),
        );
      case kRuntimeNotSelected:
        return _chooseRuntime(
          shim,
          enableGpu: enableGpu,
          enableAutomaticDownload: enableAutomaticDownload,
          timeout: timeout,
        );
    }
  }
  return _initializePlayServices(
    shim,
    enableGpu: enableGpu,
    enableAutomaticDownload: enableAutomaticDownload,
    timeout: timeout,
  );
}

/// Chooses the runtime of an auto build: Play services if both halves are
/// ready, the bundled runtime otherwise.
Future<LiteRtRuntimeStatus> _chooseRuntime(
  AndroidRuntimeShim shim, {
  required bool enableGpu,
  required bool enableAutomaticDownload,
  required Duration timeout,
}) async {
  String? error;
  try {
    final status = await _initializePlayServices(
      shim,
      enableGpu: enableGpu,
      enableAutomaticDownload: enableAutomaticDownload,
      timeout: timeout,
    );
    if (!status.interpreterReady) {
      error =
          'Google Play services could not provide the Interpreter runtime: '
          '${status.interpreterError}';
    } else if (!status.compiledModelReady) {
      error =
          'Google Play services could not provide the CompiledModel runtime: '
          '${status.compiledModelError}';
    }
  } on LiteRtRuntimeUnavailableException catch (e) {
    error =
        'Google Play services could not provide LiteRT. Interpreter: '
        '${e.status.interpreterError}. CompiledModel: '
        '${e.status.compiledModelError}';
  }
  // A concurrent call may have chosen first; its choice holds.
  final chosen = shim.select(
    error == null ? kRuntimePlayServices : kRuntimeBundled,
    playServicesError: error,
  );
  if (chosen == kRuntimePlayServices) {
    return _statusOf(shim, enableGpu: enableGpu);
  }
  return LiteRtRuntimeStatus.bundled(playServicesError: shim.playServicesError);
}

Future<LiteRtRuntimeStatus> _initializePlayServices(
  AndroidRuntimeShim shim, {
  required bool enableGpu,
  required bool enableAutomaticDownload,
  required Duration timeout,
}) {
  if (shim.interpreterState == kRuntimeStateReady &&
      shim.compiledModelState == kRuntimeStateReady) {
    return Future.value(_statusOf(shim, enableGpu: enableGpu));
  }
  final pending = _pending;
  if (pending != null) {
    return _afterPending(
      shim,
      pending,
      enableGpu: enableGpu,
      enableAutomaticDownload: enableAutomaticDownload,
      timeout: timeout,
    );
  }
  _pendingEnableGpu = enableGpu;
  _pendingEnableAutomaticDownload = enableAutomaticDownload;
  _pendingDeadline = DateTime.now().add(timeout);
  return _pending =
      _initialize(
        shim,
        enableGpu: enableGpu,
        enableAutomaticDownload: enableAutomaticDownload,
        timeout: timeout,
      ).whenComplete(() {
        _pending = null;
      });
}

/// Waits for the [pending] initialization, for at most [timeout].
///
/// Its result answers this call only if it had the same options and no less
/// time. Otherwise this call initializes again afterwards, in the time it has
/// left, since the result would answer for other options or a shorter wait.
Future<LiteRtRuntimeStatus> _afterPending(
  AndroidRuntimeShim shim,
  Future<LiteRtRuntimeStatus> pending, {
  required bool enableGpu,
  required bool enableAutomaticDownload,
  required Duration timeout,
}) async {
  final deadline = DateTime.now().add(timeout);
  final shareResult =
      _pendingEnableGpu == enableGpu &&
      _pendingEnableAutomaticDownload == enableAutomaticDownload &&
      !deadline.isAfter(_pendingDeadline!);
  try {
    final status = await pending.timeout(timeout);
    if (shareResult) return status;
  } on TimeoutException {
    final error =
        'initialization did not finish within ${timeout.inMilliseconds} ms. '
        'Google Play services may still be downloading the LiteRT module';
    return _throwIfUnavailable(
      _statusOf(
        shim,
        enableGpu: enableGpu,
        interpreterError: error,
        compiledModelError: error,
      ),
    );
  } on LiteRtRuntimeUnavailableException {
    if (shareResult) rethrow;
  }
  final remaining = deadline.difference(DateTime.now());
  return initializeAndroidRuntime(
    enableGpu: enableGpu,
    enableAutomaticDownload: enableAutomaticDownload,
    timeout: remaining.isNegative ? Duration.zero : remaining,
  );
}

LiteRtRuntimeStatus _throwIfUnavailable(LiteRtRuntimeStatus status) {
  if (!status.interpreterReady && !status.compiledModelReady) {
    throw LiteRtRuntimeUnavailableException(status);
  }
  return status;
}

Future<LiteRtRuntimeStatus> _initialize(
  AndroidRuntimeShim shim, {
  required bool enableGpu,
  required bool enableAutomaticDownload,
  required Duration timeout,
}) async {
  Map<String, Object?>? reply;
  String? pluginError;
  try {
    reply = await _channel
        .invokeMapMethod<String, Object?>('initializeRuntime', {
          'enableGpu': enableGpu,
          'enableAutomaticDownload': enableAutomaticDownload,
          'timeoutMillis': timeout.inMilliseconds,
        })
        // The plugin replies within its own timeout. This only covers a reply
        // that never comes, so that a retry is not stuck behind it.
        .timeout(timeout + _replyGrace);
  } on PlatformException catch (e) {
    pluginError =
        'the Android plugin failed to initialize LiteRT: ${e.message}';
    // A plugin bug rather than a runtime condition. An auto build goes on to
    // the bundled runtime, so make sure it gets seen.
    debugPrint('flutter_litert: $pluginError');
  } on TimeoutException {
    pluginError =
        'the Android plugin did not reply within '
        '${(timeout + _replyGrace).inSeconds} s';
  }
  if (pluginError == null) {
    // The Java and native halves are built from the same Gradle property, so
    // a mismatch means a broken build rather than a runtime condition.
    if (reply == null || reply['mode'] != 'playServices') {
      throw StateError(
        'flutter_litert: the Android plugin reported runtime '
        '${reply?['mode']}, but libflutter_litert_runtime.so was built with '
        'the Play services client. Rebuild the app.',
      );
    }
  }
  return _throwIfUnavailable(
    _statusOf(
      shim,
      enableGpu: enableGpu,
      interpreterError: pluginError ?? reply?['interpreterError'] as String?,
      compiledModelError:
          pluginError ?? reply?['compiledModelError'] as String?,
      pluginGpuError: reply?['gpuError'] as String?,
    ),
  );
}

/// Readiness is read from the shim rather than the channel reply, because the
/// shim is what the FFI bindings check.
LiteRtRuntimeStatus _statusOf(
  AndroidRuntimeShim shim, {
  required bool enableGpu,
  String? interpreterError,
  String? compiledModelError,
  String? pluginGpuError,
}) {
  final interpreterReady = shim.interpreterState == kRuntimeStateReady;
  final compiledModelReady = shim.compiledModelState == kRuntimeStateReady;
  final hasGpuAccelerator = shim.gpuAccelerator != nullptr;
  return LiteRtRuntimeStatus(
    runtime: LiteRtRuntime.playServices,
    interpreterReady: interpreterReady,
    compiledModelReady: compiledModelReady,
    gpuAvailable: compiledModelReady ? hasGpuAccelerator : null,
    interpreterError: interpreterReady
        ? null
        : interpreterError ?? shim.interpreterError,
    compiledModelError: compiledModelReady
        ? null
        : compiledModelError ?? shim.compiledModelError,
    gpuError: gpuErrorForCall(
      enableGpu: enableGpu,
      compiledModelReady: compiledModelReady,
      hasGpuAccelerator: hasGpuAccelerator,
      reason: shim.gpuError,
      pluginGpuError: pluginGpuError,
    ),
  );
}
