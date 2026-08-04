// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Verifies that a mixed Apple NPU request keeps its explicit CPU fallback.
///
/// Physical-iPhone testing found `FlutterLiteRtRegisterCoreMlNpuAccelerator`
/// returning `kLiteRtStatusErrorUnsupported`, because the Core ML framework
/// resolved through Swift Package Manager does not carry the NPU entry points.
/// Before the fix that took every `{npu, cpu}` request down with it: the caller
/// had asked for CPU fallback and got a hard failure instead.
///
/// Unlike the accuracy and timing matrix, this is meaningful on a simulator.
/// It asserts the behaviour of a registration *failure*, and a simulator
/// reproduces that condition exactly. It has no Neural Engine, so it can say
/// nothing about NPU performance or accuracy, and does not try to.
///
/// Both outcomes are asserted, so this stays valid once a patched framework
/// ships and registration starts succeeding.
///
/// Run with:
///   flutter test integration_test/apple_npu_fallback_test.dart -d `ios device`
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('a mixed Apple NPU request keeps its CPU fallback', (_) async {
    expect(
      Platform.isIOS || Platform.isMacOS,
      isTrue,
      reason: 'Core ML NPU is an Apple-platform path.',
    );

    final bytes = await _loadAsset('assets/simple_model.tflite');

    // Strict placement is the probe: it has no fallback to degrade to, so it
    // reports whether the accelerator registered at all.
    var npuAvailable = true;
    try {
      CompiledModel.fromBuffer(
        bytes,
        accelerators: const {Accelerator.npu},
      ).close();
    } catch (error) {
      npuAvailable = false;
      print('strict {npu} unavailable here: $error');
    }
    print('Core ML NPU available: $npuAvailable');

    // The regression: whichever way strict placement went, a request that
    // explicitly allows CPU must produce a usable model.
    final mixed = CompiledModel.fromBuffer(
      bytes,
      accelerators: const {Accelerator.npu, Accelerator.cpu},
    );
    addTearDown(mixed.close);

    final inputs = [
      for (final size in mixed.inputByteSizes)
        Float32List(size ~/ Float32List.bytesPerElement)
          ..fillRange(0, size ~/ Float32List.bytesPerElement, 0.5),
    ];
    final outputs = mixed.run(inputs);
    expect(outputs, isNotEmpty);
    expect(
      outputs.every((o) => o.every((v) => v.isFinite)),
      isTrue,
      reason: 'The degraded CPU path must still produce usable output.',
    );

    // `accelerators` reports what was honoured, not what was asked for, so a
    // caller can detect the narrowing without catching anything.
    if (npuAvailable) {
      expect(mixed.accelerators, contains(Accelerator.npu));
    } else {
      expect(
        mixed.accelerators,
        isNot(contains(Accelerator.npu)),
        reason: 'An unavailable NPU must be dropped from the effective set.',
      );
      expect(mixed.accelerators, contains(Accelerator.cpu));
    }
    print('effective accelerators: ${mixed.accelerators}');
  });
}

Future<Uint8List> _loadAsset(String key) async {
  final data = await rootBundle.load(key);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
