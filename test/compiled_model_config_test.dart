import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('CompiledModelConfig', () {
    test('default constructor is Auto at fp32 with managed buffers', () {
      const config = CompiledModelConfig();

      expect(config, const CompiledModelConfig.auto());
      expect(config.policy, CompiledModelPolicy.auto);
      expect(config.precision, Precision.fp32);
      expect(config.tensorBufferMode, TensorBufferMode.managed);
    });

    test('every policy exposes its exact construction sequence', () {
      const cases =
          <(CompiledModelConfig, Set<Accelerator>, Set<Accelerator>?)>[
            (CompiledModelConfig.auto(), {Accelerator.gpu}, {Accelerator.cpu}),
            (CompiledModelConfig.cpu(), {Accelerator.cpu}, null),
            (CompiledModelConfig.gpu(), {Accelerator.gpu}, null),
            (
              CompiledModelConfig.gpuWithCpuFallback(),
              {Accelerator.gpu, Accelerator.cpu},
              {Accelerator.cpu},
            ),
            (CompiledModelConfig.npu(), {Accelerator.npu}, null),
            (
              CompiledModelConfig.npuWithCpuFallback(),
              {Accelerator.npu, Accelerator.cpu},
              {Accelerator.cpu},
            ),
          ];

      for (final (config, primary, fallback) in cases) {
        expect(config.primaryAccelerators, primary, reason: config.policy.name);
        expect(
          config.fallbackAccelerators,
          fallback,
          reason: config.policy.name,
        );
        expect(
          config.allowsCpuFallback,
          fallback != null,
          reason: config.policy.name,
        );
      }
    });

    test('Auto and mixed GPU fallback remain distinct', () {
      const auto = CompiledModelConfig.auto();
      const mixed = CompiledModelConfig.gpuWithCpuFallback();

      expect(auto.primaryAccelerators, {Accelerator.gpu});
      expect(mixed.primaryAccelerators, {Accelerator.gpu, Accelerator.cpu});
      expect(auto, isNot(mixed));
    });

    test('precision and tensor buffer mode survive policy changes', () {
      const config = CompiledModelConfig.gpu(
        precision: Precision.fp16,
        tensorBufferMode: TensorBufferMode.hostMemory,
      );
      final copy = config.copyWith(policy: CompiledModelPolicy.npu);

      expect(copy.policy, CompiledModelPolicy.npu);
      expect(copy.precision, Precision.fp16);
      expect(copy.tensorBufferMode, TensorBufferMode.hostMemory);
    });

    test('has value equality, hash code, and diagnostic text', () {
      const first = CompiledModelConfig.gpu(precision: Precision.fp16);
      const second = CompiledModelConfig(
        policy: CompiledModelPolicy.gpu,
        precision: Precision.fp16,
      );

      expect(first, second);
      expect(first.hashCode, second.hashCode);
      expect(first.toString(), contains('CompiledModelPolicy.gpu'));
      expect(first.toString(), contains('Precision.fp16'));
      expect(first.toString(), contains('TensorBufferMode.managed'));
    });

    test('accelerator sequences cannot be mutated', () {
      const config = CompiledModelConfig.auto();

      expect(
        () => config.primaryAccelerators.add(Accelerator.npu),
        throwsUnsupportedError,
      );
      expect(
        () => config.fallbackAccelerators!.add(Accelerator.gpu),
        throwsUnsupportedError,
      );
    });
  });
}
