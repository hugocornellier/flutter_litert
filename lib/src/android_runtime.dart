/// Where an Android app gets its LiteRT runtime from.
///
/// This is chosen at build time with the `flutterLitert.runtime` Gradle
/// property, or, with `flutterLitert.runtime=auto`, by the first
/// `initializeAndroidRuntime` call. Every other platform always uses
/// [bundled].
enum LiteRtRuntime {
  /// LiteRT is packaged in the app (the default, and the fallback of `auto`).
  bundled,

  /// LiteRT comes from Google Play services (`flutterLitert.runtime=playServices`,
  /// or `auto` on a device where Play services provides it).
  ///
  /// The app must call `initializeAndroidRuntime` before it creates an
  /// `Interpreter` or a `CompiledModel`.
  playServices,
}

/// The outcome of `initializeAndroidRuntime`.
///
/// The Google Play services runtime has two independent halves: the classic
/// TFLite runtime behind `Interpreter`, and the LiteRT runtime behind
/// `CompiledModel`. A device can provide one without the other, so each is
/// reported on its own.
final class LiteRtRuntimeStatus {
  const LiteRtRuntimeStatus({
    required this.runtime,
    required this.interpreterReady,
    required this.compiledModelReady,
    this.gpuAvailable,
    this.interpreterError,
    this.compiledModelError,
    this.gpuError,
    this.playServicesError,
  });

  /// Status of the [LiteRtRuntime.bundled] runtime, which is always ready.
  const LiteRtRuntimeStatus.bundled({this.playServicesError})
    : runtime = LiteRtRuntime.bundled,
      interpreterReady = true,
      compiledModelReady = true,
      gpuAvailable = null,
      interpreterError = null,
      compiledModelError = null,
      gpuError = null;

  /// The runtime in use.
  final LiteRtRuntime runtime;

  /// Whether `Interpreter` can be used.
  final bool interpreterReady;

  /// Whether `CompiledModel` can be used.
  final bool compiledModelReady;

  /// Whether the Play services runtime provided a GPU accelerator for
  /// `CompiledModel`.
  ///
  /// Null when this is not known up front: with the bundled runtime, and with
  /// Play services before the CompiledModel runtime is ready. A GPU request is
  /// then resolved when the model is compiled, as usual.
  final bool? gpuAvailable;

  /// Why the Interpreter runtime is unavailable, when it is.
  final String? interpreterError;

  /// Why the CompiledModel runtime is unavailable, when it is.
  final String? compiledModelError;

  /// Why GPU support was requested but not initialized, when it was not.
  ///
  /// The CompiledModel runtime then falls back to CPU only, so this can be set
  /// while [compiledModelReady] is true.
  final String? gpuError;

  /// Why a `flutterLitert.runtime=auto` build uses the bundled runtime rather
  /// than Google Play services, when it does.
  ///
  /// Null in isolates other than the one that made the choice.
  final String? playServicesError;

  /// Whether both `Interpreter` and `CompiledModel` can be used.
  bool get isReady => interpreterReady && compiledModelReady;

  @override
  String toString() =>
      'LiteRtRuntimeStatus(runtime: ${runtime.name}, '
      'interpreterReady: $interpreterReady, '
      'compiledModelReady: $compiledModelReady, '
      'gpuAvailable: $gpuAvailable'
      '${interpreterError == null ? '' : ', interpreterError: $interpreterError'}'
      '${compiledModelError == null ? '' : ', compiledModelError: $compiledModelError'}'
      '${gpuError == null ? '' : ', gpuError: $gpuError'}'
      '${playServicesError == null ? '' : ', playServicesError: $playServicesError'})';
}

/// Thrown by `initializeAndroidRuntime` when neither half of the Google Play
/// services runtime could be initialized.
///
/// Typical causes are a device without Google Play services, or a download of
/// the runtime module that failed or was not allowed. Calling
/// `initializeAndroidRuntime` again retries.
final class LiteRtRuntimeUnavailableException implements Exception {
  const LiteRtRuntimeUnavailableException(this.status);

  /// The status, including both failure reasons.
  final LiteRtRuntimeStatus status;

  @override
  String toString() =>
      'LiteRtRuntimeUnavailableException: the Google Play services LiteRT '
      'runtime is unavailable. Interpreter: ${status.interpreterError}. '
      'CompiledModel: ${status.compiledModelError}.';
}
