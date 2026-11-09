/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// Option types shared by every [CompiledModel] platform implementation
/// (native FFI, web LiteRT.js, unsupported stub). Each implementation file
/// re-exports these so the conditional export in `compiled_model.dart`
/// resolves the same names everywhere.
library;

/// Hardware accelerator requested for LiteRT Next compilation.
///
/// On the web, [cpu] maps to the LiteRT.js WASM backend and [gpu] to WebGPU.
/// [npu] uses Core ML's CPU-and-Neural-Engine policy on iOS 16 or newer and
/// Apple Silicon Macs running macOS 14 or newer. On Android, [npu] uses a
/// packaged vendor runtime on supported API 31+ arm64 devices.
enum Accelerator { cpu, gpu, npu }

/// GPU precision mode for LiteRT Next compilation.
enum Precision { fp16, fp32 }

/// Tensor buffer allocation mode for CompiledModel I/O.
///
/// [managed] uses LiteRT-managed buffers and copies host data through the
/// documented lock/write/unlock and lock/read/unlock path. [hostMemory] wraps
/// package-owned, 64-byte-aligned host memory with
/// `LiteRtCreateTensorBufferFromHostMemory`.
///
/// Host-memory buffers are opt-in because performance is model- and
/// accelerator-dependent: they can reduce lock/copy overhead for some strict
/// GPU models, but are slower for others. [hostMemory] is native-only; the web
/// implementation supports [managed] buffers exclusively.
enum TensorBufferMode { managed, hostMemory }

/// Ordered accelerator policy for building a [CompiledModel].
///
/// This is deliberately separate from [Accelerator]. An accelerator is a
/// hardware target, while a policy describes preference and fallback order;
/// an unordered `Set<Accelerator>` cannot distinguish an automatic choice
/// from an exact request.
enum CompiledModelPolicy {
  /// Package-recommended selection.
  ///
  /// In flutter_litert 3.9 this means a strict GPU request at
  /// [Precision.fp32] (on the web, at LiteRT.js's default precision unless
  /// one is set), with a complete CPU-only retry when GPU construction
  /// fails. It does not first request a mixed `{gpu, cpu}` graph. NPU is
  /// intentionally excluded because availability and correctness remain
  /// device- and model-specific.
  ///
  /// Auto is a construction policy, not a correctness test or benchmark. A
  /// model that compiles can still be numerically unsuitable for a backend;
  /// validate production models separately with `verifyCompiledModel`.
  auto,

  /// CPU only.
  cpu,

  /// GPU compilation with no complete CPU retry.
  ///
  /// "Strict" here describes the requested accelerator set and fallback
  /// policy. Use `CompiledModel.isFullyAccelerated` to inspect whether the
  /// runtime placed the whole graph on an accelerator.
  gpu,

  /// Request a mixed `{gpu, cpu}` graph, then retry the whole model CPU-only
  /// if construction fails.
  gpuWithCpuFallback,

  /// NPU compilation with no complete CPU retry.
  npu,

  /// Request a mixed `{npu, cpu}` graph, then retry the whole model CPU-only
  /// if construction fails.
  ///
  /// NPU is experimental and must be validated per model and target device.
  npuWithCpuFallback,
}

/// Value configuration for policy-based [CompiledModel] construction.
///
/// Use this with `CompiledModel.fromFileWithConfig`,
/// `CompiledModel.fromBufferWithConfig`, or
/// `CompiledModel.fromBufferWithConfigAsync`. The existing low-level
/// constructors that accept an exact `Set<Accelerator>` remain available for
/// advanced combinations and retain their original semantics.
class CompiledModelConfig {
  /// Ordered accelerator selection and fallback policy.
  final CompiledModelPolicy policy;

  /// GPU precision: [requestedPrecision], or [Precision.fp32] when the
  /// caller left it unset.
  ///
  /// This is retained for CPU and NPU policies so a single immutable config
  /// can move between policies without losing the user's precision choice.
  /// Those backends do not currently expose a configurable precision.
  Precision get precision => _precision ?? Precision.fp32;

  /// The precision the caller passed, or null when it was left unset.
  ///
  /// Native platforms compile an unset precision as [Precision.fp32]. On the
  /// web, an unset precision leaves WebGPU at LiteRT.js's own default, which
  /// is fp16 when the GPU supports `shader-f16` and fp32 otherwise.
  Precision? get requestedPrecision => _precision;

  final Precision? _precision;

  /// Tensor buffer allocation mode.
  final TensorBufferMode tensorBufferMode;

  /// Creates a policy-based CompiledModel configuration.
  const CompiledModelConfig({
    this.policy = CompiledModelPolicy.auto,
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : _precision = precision;

  /// Package-recommended accelerator selection.
  const CompiledModelConfig.auto({
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.auto,
       _precision = precision;

  /// CPU-only compilation.
  const CompiledModelConfig.cpu({
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.cpu,
       _precision = precision;

  /// Strict GPU compilation.
  const CompiledModelConfig.gpu({
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.gpu,
       _precision = precision;

  /// GPU-preferred compilation with a complete CPU retry.
  const CompiledModelConfig.gpuWithCpuFallback({
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.gpuWithCpuFallback,
       _precision = precision;

  /// Strict NPU compilation.
  const CompiledModelConfig.npu({
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.npu,
       _precision = precision;

  /// NPU-preferred compilation with a complete CPU retry.
  const CompiledModelConfig.npuWithCpuFallback({
    Precision? precision,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.npuWithCpuFallback,
       _precision = precision;

  /// Exact accelerator set used for the policy's first construction attempt.
  Set<Accelerator> get primaryAccelerators => switch (policy) {
    CompiledModelPolicy.auto ||
    CompiledModelPolicy.gpu => const {Accelerator.gpu},
    CompiledModelPolicy.cpu => const {Accelerator.cpu},
    CompiledModelPolicy.gpuWithCpuFallback => const {
      Accelerator.gpu,
      Accelerator.cpu,
    },
    CompiledModelPolicy.npu => const {Accelerator.npu},
    CompiledModelPolicy.npuWithCpuFallback => const {
      Accelerator.npu,
      Accelerator.cpu,
    },
  };

  /// Whether a failed preferred compilation is retried as CPU-only.
  bool get allowsCpuFallback => switch (policy) {
    CompiledModelPolicy.auto ||
    CompiledModelPolicy.gpuWithCpuFallback ||
    CompiledModelPolicy.npuWithCpuFallback => true,
    _ => false,
  };

  /// Accelerator set used for a complete retry after primary construction
  /// fails, or null when the policy has no complete fallback.
  Set<Accelerator>? get fallbackAccelerators =>
      allowsCpuFallback ? const {Accelerator.cpu} : null;

  /// Returns a copy with the supplied fields replaced.
  CompiledModelConfig copyWith({
    CompiledModelPolicy? policy,
    Precision? precision,
    TensorBufferMode? tensorBufferMode,
  }) {
    return CompiledModelConfig(
      policy: policy ?? this.policy,
      precision: precision ?? _precision,
      tensorBufferMode: tensorBufferMode ?? this.tensorBufferMode,
    );
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is CompiledModelConfig &&
            policy == other.policy &&
            _precision == other._precision &&
            tensorBufferMode == other.tensorBufferMode;
  }

  @override
  int get hashCode => Object.hash(policy, _precision, tensorBufferMode);

  @override
  String toString() {
    final requested = _precision;
    final precisionText = requested == null
        ? 'unset'
        : 'Precision.${requested.name}';
    return 'CompiledModelConfig('
        'policy: CompiledModelPolicy.${policy.name}, '
        'precision: $precisionText, '
        'tensorBufferMode: TensorBufferMode.${tensorBufferMode.name})';
  }
}
