import 'dart:async';
import 'dart:ffi';

import 'package:flutter/services.dart';

import '../android_runtime.dart';
import '../bindings/android_runtime_shim.dart';

const MethodChannel _channel = MethodChannel('flutter_litert');

Future<LiteRtRuntimeStatus>? _pending;
bool? _pendingEnableGpu;
bool? _pendingEnableAutomaticDownload;

/// Why the CompiledModel runtime has no GPU accelerator, recorded by the call
/// that initialized it, so that later calls asking for the GPU report it too.
String? _gpuReason;

/// How much longer than the plugin's own timeout to wait for its reply.
const Duration _replyGrace = Duration(seconds: 30);

/// The LiteRT runtime this app was built with.
///
/// [LiteRtRuntime.playServices] only on Android apps built with
/// `flutterLitert.runtime=playServices`; [LiteRtRuntime.bundled] otherwise.
LiteRtRuntime get androidRuntime => usesPlayServicesRuntime
    ? LiteRtRuntime.playServices
    : LiteRtRuntime.bundled;

/// Initializes the Google Play services LiteRT runtime.
///
/// Apps built with `flutterLitert.runtime=playServices` must await this, from
/// the main isolate, before creating an `Interpreter` or a `CompiledModel`.
/// Isolates spawned afterwards can use both without calling it again. With the
/// bundled runtime, and on every platform but Android, it returns
/// [LiteRtRuntimeStatus.bundled] immediately.
///
/// [enableGpu] asks Play services for its GPU accelerator, for
/// `CompiledModel` with `Accelerator.gpu`. If the GPU request fails, the
/// CompiledModel runtime is initialized for CPU only and
/// [LiteRtRuntimeStatus.gpuError] says why. The first successful
/// initialization fixes this for the process: asking for the GPU later does
/// not add it, and the status of that later call says so.
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
/// with other options runs after it. A runtime half that failed is retried by
/// the next call. Throws [LiteRtRuntimeUnavailableException] when neither half
/// could be initialized; when only one could, the returned status says which
/// and why.
Future<LiteRtRuntimeStatus> initializeAndroidRuntime({
  bool enableGpu = false,
  bool enableAutomaticDownload = true,
  Duration timeout = const Duration(minutes: 5),
}) {
  final shim = androidRuntimeShim;
  if (shim == null || !shim.isPlayServices) {
    return Future.value(const LiteRtRuntimeStatus.bundled());
  }
  if (shim.interpreterState == kRuntimeStateReady &&
      shim.compiledModelState == kRuntimeStateReady) {
    return Future.value(_statusOf(shim, enableGpu: enableGpu));
  }
  final pending = _pending;
  if (pending != null) {
    if (_pendingEnableGpu == enableGpu &&
        _pendingEnableAutomaticDownload == enableAutomaticDownload) {
      return pending;
    }
    // Joining would answer with the pending call's options, so run after it.
    return pending
        .then<void>((_) {}, onError: (Object _) {})
        .then(
          (_) => initializeAndroidRuntime(
            enableGpu: enableGpu,
            enableAutomaticDownload: enableAutomaticDownload,
            timeout: timeout,
          ),
        );
  }
  _pendingEnableGpu = enableGpu;
  _pendingEnableAutomaticDownload = enableAutomaticDownload;
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
        '${reply?['mode']}, but libflutter_litert_runtime.so was built for '
        'playServices. Rebuild the app.',
      );
    }
    if (reply['compiledModelAttached'] == true) {
      _gpuReason = gpuUnavailableReason(
        gpuRequested: enableGpu,
        pluginGpuError: reply['gpuError'] as String?,
      );
    }
  }
  final status = _statusOf(
    shim,
    enableGpu: enableGpu,
    interpreterError: pluginError ?? reply?['interpreterError'] as String?,
    compiledModelError: pluginError ?? reply?['compiledModelError'] as String?,
    pluginGpuError: reply?['gpuError'] as String?,
  );
  if (!status.interpreterReady && !status.compiledModelReady) {
    throw LiteRtRuntimeUnavailableException(status);
  }
  return status;
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
      reason: _gpuReason,
      pluginGpuError: pluginGpuError,
    ),
  );
}
