import 'package:flutter_litert/src/web/accelerator_policy.dart';
import 'package:flutter_test/flutter_test.dart';

/// Pins the per-runner backend aggregation rule. A live browser only exercises
/// the split it happens to produce, so every ordering is checked here,
/// including the one that breaks a naive first-runner rule.
void main() {
  group('aggregateActiveAccelerator', () {
    test('all runners on webgpu reports webgpu', () {
      expect(
        aggregateActiveAccelerator(const <String?>['webgpu', 'webgpu']),
        'webgpu',
      );
    });

    test('all runners on wasm reports wasm', () {
      expect(
        aggregateActiveAccelerator(const <String?>['wasm', 'wasm', 'wasm']),
        'wasm',
      );
    });

    test('a later runner on webgpu still reports webgpu', () {
      // The failure ordering for a first-non-null rule: the first runner fell
      // back to WASM while a later one stayed on WebGPU.
      expect(
        aggregateActiveAccelerator(const <String?>['wasm', 'webgpu', 'wasm']),
        'webgpu',
      );
    });

    test('the Chrome-observed split (first on webgpu, later on wasm)', () {
      expect(
        aggregateActiveAccelerator(const <String?>[
          'webgpu',
          'webgpu',
          'webgpu',
          'wasm',
          null,
        ]),
        'webgpu',
      );
    });

    test('nulls are skipped for the first-non-null fallback', () {
      expect(
        aggregateActiveAccelerator(const <String?>[null, null, 'wasm', null]),
        'wasm',
      );
    });

    test('all null (pre-init) reports null', () {
      expect(
        aggregateActiveAccelerator(const <String?>[null, null, null]),
        isNull,
      );
    });

    test('empty reports null', () {
      expect(aggregateActiveAccelerator(const <String?>[]), isNull);
    });
  });
}
