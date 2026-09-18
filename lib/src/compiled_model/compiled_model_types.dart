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
/// [npu] uses Core ML's CPU-and-Neural-Engine policy on iOS 13 or newer and
/// Apple Silicon Macs running macOS 13 or newer. On Android, [npu] uses a
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
  /// [Precision.fp32], with a complete CPU-only retry when GPU construction
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

  /// GPU precision.
  ///
  /// This is retained for CPU and NPU policies so a single immutable config
  /// can move between policies without losing the user's precision choice.
  /// Those backends do not currently expose a configurable precision.
  final Precision precision;

  /// Tensor buffer allocation mode.
  final TensorBufferMode tensorBufferMode;

  /// Creates a policy-based CompiledModel configuration.
  const CompiledModelConfig({
    this.policy = CompiledModelPolicy.auto,
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  });

  /// Package-recommended accelerator selection.
  const CompiledModelConfig.auto({
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.auto;

  /// CPU-only compilation.
  const CompiledModelConfig.cpu({
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.cpu;

  /// Strict GPU compilation.
  const CompiledModelConfig.gpu({
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.gpu;

  /// GPU-preferred compilation with a complete CPU retry.
  const CompiledModelConfig.gpuWithCpuFallback({
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.gpuWithCpuFallback;

  /// Strict NPU compilation.
  const CompiledModelConfig.npu({
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.npu;

  /// NPU-preferred compilation with a complete CPU retry.
  const CompiledModelConfig.npuWithCpuFallback({
    this.precision = Precision.fp32,
    this.tensorBufferMode = TensorBufferMode.managed,
  }) : policy = CompiledModelPolicy.npuWithCpuFallback;

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
      precision: precision ?? this.precision,
      tensorBufferMode: tensorBufferMode ?? this.tensorBufferMode,
    );
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is CompiledModelConfig &&
            policy == other.policy &&
            precision == other.precision &&
            tensorBufferMode == other.tensorBufferMode;
  }

  @override
  int get hashCode => Object.hash(policy, precision, tensorBufferMode);

  @override
  String toString() {
    return 'CompiledModelConfig('
        'policy: CompiledModelPolicy.${policy.name}, '
        'precision: Precision.${precision.name}, '
        'tensorBufferMode: TensorBufferMode.${tensorBufferMode.name})';
  }
}
