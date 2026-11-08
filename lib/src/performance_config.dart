/// Hardware acceleration mode for LiteRT inference.
enum PerformanceMode {
  /// No hardware acceleration.
  disabled,

  /// XNNPACK delegate (CPU-optimized, all native platforms).
  xnnpack,

  /// Metal GPU delegate on iOS and macOS.
  ///
  /// Android has no Interpreter GPU delegate since flutter_litert 4.0.0, so
  /// this mode uses XNNPACK there (with a one-time warning), and no delegate on
  /// the Play services runtime. For the GPU on Android, use `CompiledModel`
  /// with `Accelerator.gpu`. Other platforms run on CPU.
  gpu,

  /// CoreML delegate (iOS/macOS only).
  coreml,

  /// Use the platform mapping implemented by `InterpreterFactory`.
  auto,
}

/// Configuration for interpreter hardware acceleration and threading.
class PerformanceConfig {
  /// The hardware acceleration mode.
  final PerformanceMode mode;

  /// Number of threads for inference. Defaults to min(4, CPU count) if null.
  final int? numThreads;

  /// Creates a performance config with the given [mode] and optional [numThreads].
  const PerformanceConfig({this.mode = PerformanceMode.auto, this.numThreads});

  /// Creates an XNNPACK-accelerated config.
  const PerformanceConfig.xnnpack({this.numThreads})
    : mode = PerformanceMode.xnnpack;

  /// Creates a GPU-accelerated config: Metal on iOS and macOS, XNNPACK on
  /// Android. See [PerformanceMode.gpu].
  const PerformanceConfig.gpu({this.numThreads}) : mode = PerformanceMode.gpu;

  /// Creates a CoreML-accelerated config.
  const PerformanceConfig.coreml({this.numThreads})
    : mode = PerformanceMode.coreml;

  /// Creates a config that uses `InterpreterFactory`'s auto-mode mapping.
  const PerformanceConfig.auto({this.numThreads}) : mode = PerformanceMode.auto;

  /// A config with no hardware acceleration.
  static const PerformanceConfig disabled = PerformanceConfig(
    mode: PerformanceMode.disabled,
  );
}
