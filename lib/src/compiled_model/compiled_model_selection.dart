import 'dart:typed_data';

import 'compiled_model.dart';

/// Builds a [CompiledModel] from [bytes], choosing between the GPU-with-CPU
/// fallback path and an explicit accelerator set.
///
/// Detectors typically expose an `accelerators` knob that defaults to
/// `{Accelerator.gpu, Accelerator.cpu}`, meaning "try GPU, fall back to CPU".
/// That default has to route through [CompiledModel.fromBufferWithGpuFallback],
/// because [CompiledModel.fromBuffer] treats the set as a hard requirement and
/// does not degrade. Any other set is an explicit request and goes to
/// [CompiledModel.fromBuffer] unchanged.
///
/// This helper centralizes that branch so callers do not each re-derive the
/// "is this the default set?" predicate.
///
/// [onGpuFallback] is invoked with the compile error when the GPU attempt fails
/// and the model falls back to CPU. [forceCpu] skips the GPU attempt entirely.
CompiledModel compiledModelFromBufferAuto(
  Uint8List bytes, {
  Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
  Precision precision = Precision.fp16,
  TensorBufferMode tensorBufferMode = TensorBufferMode.managed,
  bool forceCpu = false,
  void Function(Object error)? onGpuFallback,
}) {
  // Handled before the accelerator branch so forceCpu wins regardless of the
  // requested set, including explicit sets that never reach the fallback path.
  if (forceCpu) {
    return CompiledModel.fromBuffer(
      bytes,
      accelerators: const {Accelerator.cpu},
      precision: precision,
      tensorBufferMode: tensorBufferMode,
    );
  }
  if (isDefaultGpuCpuAccelerators(accelerators)) {
    return CompiledModel.fromBufferWithGpuFallback(
      bytes,
      precision: precision,
      tensorBufferMode: tensorBufferMode,
      onFallback: onGpuFallback,
    );
  }
  return CompiledModel.fromBuffer(
    bytes,
    accelerators: accelerators,
    precision: precision,
    tensorBufferMode: tensorBufferMode,
  );
}

/// Whether [accelerators] is exactly the default "try GPU, fall back to CPU"
/// set of `{Accelerator.gpu, Accelerator.cpu}`.
///
/// Callers use this to distinguish the permissive default from an explicit
/// accelerator request that must be honoured as-is.
bool isDefaultGpuCpuAccelerators(Set<Accelerator> accelerators) {
  return accelerators.length == 2 &&
      accelerators.contains(Accelerator.gpu) &&
      accelerators.contains(Accelerator.cpu);
}
