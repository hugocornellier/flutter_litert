// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Physical-device gate for the Android LiteRT Next OpenCL/GL accelerator.
///
/// This suite intentionally has no skip. It must only be sent to a physical
/// arm64 Android device. The known-output model must compile and run with
/// strict {gpu} at both fp32 and fp16 (fp16 is the example app's default
/// precision): any accelerator registration, compilation, driver, or
/// inference failure there fails the Test Lab matrix. The
/// fromBufferWithGpuFallback factory must complete without falling back,
/// pinning that real hardware gets real GPU. The same strict-{gpu} pattern
/// must also work entirely inside a worker isolate, the architecture the
/// example app ships. The convolutional model also
/// tries strict {gpu} first, but tolerates exactly LiteRT's documented
/// gpu-only compilation refusal (LiteRtStatus=504, an op-coverage gap
/// observed with MobileFaceNet on Pixel 7) by validating the {gpu, cpu}
/// production placement instead; every other failure stays fatal. Its
/// outputs are compared element-wise against a CPU-compiled reference, so
/// finite-but-wrong GPU arithmetic fails the gate.
///
/// Inference uses the synchronous [CompiledModel.run] on purpose: runAsync
/// dispatches from a helper isolate thread, and thread-affine Android GL/CL
/// driver stacks are documented as unvalidated with it. This gate proves the
/// supported path first; runAsync validation can be layered on once green.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('strict GPU compiles and runs on physical Android hardware', (
    tester,
  ) async {
    expect(
      Platform.isAndroid,
      isTrue,
      reason: 'This gate must only run on Android.',
    );

    final device = await DeviceInfoPlugin().androidInfo;
    print(
      'Test Lab device: ${device.manufacturer} ${device.model}; '
      'SDK ${device.version.sdkInt}; hardware=${device.hardware}; '
      'ABIs=${device.supportedAbis.join(',')}',
    );
    expect(
      device.isPhysicalDevice,
      isTrue,
      reason: 'Virtual devices do not validate vendor OpenCL drivers.',
    );
    expect(
      device.supportedAbis,
      contains('arm64-v8a'),
      reason: 'The physical GPU gate requires the arm64 accelerator binary.',
    );

    await _verifyKnownOutput(precision: Precision.fp32);
    await _verifyKnownOutput(precision: Precision.fp16);
    await _verifyFallbackFactoryStaysOnGpu();
    await _verifyWorkerIsolatePattern();
    await _verifyConvolutionalModel();
  });
}

Future<Uint8List> _loadAsset(String path) async {
  final data = await rootBundle.load(path);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

Future<void> _verifyKnownOutput({required Precision precision}) async {
  final model = CompiledModel.fromBuffer(
    await _loadAsset('assets/simple_model.tflite'),
    accelerators: const {Accelerator.gpu},
    precision: precision,
  );
  try {
    expect(model.accelerators, const {Accelerator.gpu});
    final input = Float32List(model.inputByteSizes.single ~/ 4)..first = 3;
    final output = model.run([input]);
    expect(output, hasLength(1));
    // y = 2x + 1 is exactly representable at both precisions; the loose
    // fp16 tolerance only absorbs accumulation-order differences.
    final tolerance = precision == Precision.fp16 ? 1e-2 : 1e-3;
    expect(
      output.single.first,
      closeTo(7, tolerance),
      reason: 'Strict GPU known-output check failed at ${precision.name}.',
    );
    print('Strict GPU known-output model verified at ${precision.name}.');
  } finally {
    model.close();
  }
}

/// The production convenience factory must not fall back on real hardware:
/// the known-output model is fully GPU-compatible (proven by the strict
/// {gpu} legs above), so an onFallback invocation here means a regression
/// silently degraded physical devices to CPU.
Future<void> _verifyFallbackFactoryStaysOnGpu() async {
  Object? fallbackError;
  final model = CompiledModel.fromBufferWithGpuFallback(
    await _loadAsset('assets/simple_model.tflite'),
    onFallback: (error) => fallbackError = error,
  );
  try {
    expect(
      fallbackError,
      isNull,
      reason:
          'fromBufferWithGpuFallback fell back to CPU on physical '
          'hardware: $fallbackError',
    );
    expect(model.accelerators, const {Accelerator.gpu, Accelerator.cpu});
    final input = Float32List(model.inputByteSizes.single ~/ 4)..first = 3;
    expect(model.run([input]).single.first, closeTo(7, 1e-3));
    print('GPU fallback factory compiled without falling back.');
  } finally {
    model.close();
  }
}

/// The example app compiles, runs, and closes CompiledModel entirely inside
/// a worker isolate, which executes on different OS threads than the
/// platform thread and may migrate threads across suspension points. GL/CL
/// driver stacks are thread-affine, so validate that exact production
/// pattern on hardware: strict {gpu} compile, synchronous run (as the
/// example uses), and close, all within a spawned isolate that then exits
/// with its LiteRT environment still alive (mirroring the example's
/// engine-switch dispose).
Future<void> _verifyWorkerIsolatePattern() async {
  final bytes = await _loadAsset('assets/simple_model.tflite');
  final double result = await Isolate.run(() {
    final model = CompiledModel.fromBuffer(
      bytes,
      accelerators: const {Accelerator.gpu},
      precision: Precision.fp32,
    );
    try {
      final input = Float32List(model.inputByteSizes.single ~/ 4)..first = 3;
      return model.run([input]).single.first;
    } finally {
      model.close();
    }
  });
  expect(
    result,
    closeTo(7, 1e-3),
    reason: 'Strict GPU inference inside a worker isolate returned $result.',
  );
  print('Worker-isolate strict GPU compile/run/close verified.');
}

Future<void> _verifyConvolutionalModel() async {
  final bytes = await _loadAsset('assets/mobilefacenet.tflite');

  var accelerators = const {Accelerator.gpu};
  final compileWatch = Stopwatch()..start();
  CompiledModel model;
  try {
    model = CompiledModel.fromBuffer(
      bytes,
      accelerators: accelerators,
      precision: Precision.fp32,
    );
  } on StateError catch (e) {
    // 504 is LiteRT refusing gpu-only compilation because an op has no GPU
    // implementation. The known-output model above already proved the GPU
    // stack itself, so downgrade only this exact case to the {gpu, cpu}
    // production placement. Anything else (driver crash, registration
    // failure) must still fail the gate.
    if (!e.message.toString().contains('LiteRtStatus=504')) rethrow;
    print(
      'Strict GPU MobileFaceNet compilation refused (504, unsupported op); '
      'validating the {gpu, cpu} production placement instead.',
    );
    accelerators = const {Accelerator.gpu, Accelerator.cpu};
    model = CompiledModel.fromBuffer(
      bytes,
      accelerators: accelerators,
      precision: Precision.fp32,
    );
  }
  compileWatch.stop();

  try {
    expect(model.accelerators, accelerators);
    final inputs = <Float32List>[
      for (final byteSize in model.inputByteSizes) Float32List(byteSize ~/ 4),
    ];
    for (final input in inputs) {
      for (var i = 0; i < input.length; i++) {
        input[i] = (i % 17) / 16;
      }
    }

    final inferenceWatch = Stopwatch()..start();
    final outputs = model.run(inputs);
    inferenceWatch.stop();

    expect(outputs, hasLength(model.outputCount));
    for (var i = 0; i < outputs.length; i++) {
      expect(outputs[i], hasLength(model.outputByteSizes[i] ~/ 4));
      expect(outputs[i], isNotEmpty);
      expect(
        outputs[i].every((value) => value.isFinite),
        isTrue,
        reason: 'GPU output $i contains NaN or infinity.',
      );
    }

    // Finite is not correct: compare against a CPU-compiled reference so
    // wrong-but-finite GPU arithmetic fails the gate. The tolerance
    // (atol 1e-2 + rtol 1e-2) absorbs fp accumulation-order differences
    // between backends while still rejecting garbage.
    final cpuModel = CompiledModel.fromBuffer(
      bytes,
      accelerators: const {Accelerator.cpu},
    );
    try {
      final cpuOutputs = cpuModel.run(inputs);
      expect(cpuOutputs, hasLength(outputs.length));
      var maxDiff = 0.0;
      for (var i = 0; i < outputs.length; i++) {
        expect(outputs[i], hasLength(cpuOutputs[i].length));
        for (var j = 0; j < outputs[i].length; j++) {
          final diff = (outputs[i][j] - cpuOutputs[i][j]).abs();
          if (diff > maxDiff) maxDiff = diff;
          final allowed = 1e-2 + 1e-2 * cpuOutputs[i][j].abs();
          if (!(diff <= allowed)) {
            fail(
              'GPU output $i[$j] = ${outputs[i][j]} diverges from the CPU '
              'reference ${cpuOutputs[i][j]} (diff $diff > $allowed).',
            );
          }
        }
      }
      print(
        'MobileFaceNet GPU output matches the CPU reference '
        '(max |diff| = $maxDiff).',
      );
    } finally {
      cpuModel.close();
    }

    print(
      'Strict GPU MobileFaceNet: compile='
      '${compileWatch.elapsedMilliseconds}ms, inference='
      '${inferenceWatch.elapsedMilliseconds}ms',
    );
  } finally {
    model.close();
  }
}
