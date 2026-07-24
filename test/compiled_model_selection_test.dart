import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// [isDefaultGpuCpuAccelerators] is the predicate that decides whether an
/// accelerator set is the permissive "try GPU, fall back to CPU" default (which
/// must route through the GPU-fallback constructor) or an explicit request
/// (which must be honoured as-is).
///
/// The full [compiledModelFromBufferAuto] path needs a real model buffer and a
/// platform runtime, so it is exercised by the detector suites; the branch
/// predicate is pinned here.
void main() {
  group('isDefaultGpuCpuAccelerators', () {
    test('the default gpu+cpu set is the permissive default', () {
      expect(
        isDefaultGpuCpuAccelerators(const {Accelerator.gpu, Accelerator.cpu}),
        isTrue,
      );
    });

    test('set order does not matter', () {
      expect(
        isDefaultGpuCpuAccelerators(const {Accelerator.cpu, Accelerator.gpu}),
        isTrue,
      );
    });

    test('cpu alone is an explicit request', () {
      expect(isDefaultGpuCpuAccelerators(const {Accelerator.cpu}), isFalse);
    });

    test('gpu alone is an explicit request', () {
      expect(isDefaultGpuCpuAccelerators(const {Accelerator.gpu}), isFalse);
    });

    test('empty is not the default', () {
      expect(isDefaultGpuCpuAccelerators(const <Accelerator>{}), isFalse);
    });

    test('a superset containing gpu+cpu is not the default', () {
      // A third accelerator makes this an explicit request, not the two-way
      // default, so it must not silently route through the fallback path.
      expect(
        isDefaultGpuCpuAccelerators(const {
          Accelerator.gpu,
          Accelerator.cpu,
          Accelerator.npu,
        }),
        isFalse,
      );
    });
  });
}
