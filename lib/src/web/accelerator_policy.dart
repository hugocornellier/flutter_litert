/// Aggregates the per-runner web backends of a multi-model detector into the
/// single accelerator it should report.
///
/// A detector that compiles several model runners (a detection stage, a
/// landmark stage, and so on) compiles each one independently, and any runner
/// can fall back from WebGPU to WASM on its own when the browser rejects an op.
/// A detector can therefore hold a mix of both backends at once. This has been
/// observed live on Chrome, where one model lands on WASM while the others stay
/// on WebGPU.
///
/// Returns `'webgpu'` when ANY runner is still on WebGPU. The runtime GPU-error
/// fallback and the slow-WebGPU warmup in the `WebGpuFallback` mixin are both
/// gated on the accelerator a detector reports, so a detector with even one
/// WebGPU runner must report `'webgpu'` to keep those paths armed. Reporting
/// the first runner's backend instead disables them whenever the first runner
/// fell back to WASM but a later one stayed on WebGPU.
///
/// Falls back to the first non-null backend when no runner is on WebGPU, and
/// returns null only when every runner is still uninitialized.
///
/// The [backends] are each runner's `activeAccelerator` (`'webgpu'` / `'wasm'`,
/// or null before that runner has initialized).
String? aggregateActiveAccelerator(Iterable<String?> backends) {
  String? firstNonNull;
  for (final String? backend in backends) {
    if (backend == 'webgpu') return 'webgpu';
    firstNonNull ??= backend;
  }
  return firstNonNull;
}
