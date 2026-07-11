// ignore_for_file: avoid_print

import 'dart:io';
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
/// strict {gpu}: any accelerator registration, compilation, driver, or
/// inference failure there fails the Test Lab matrix. The convolutional
/// model also tries strict {gpu} first, but tolerates exactly LiteRT's
/// documented gpu-only compilation refusal (LiteRtStatus=504, an op-coverage
/// gap observed with MobileFaceNet on Pixel 7) by validating the {gpu, cpu}
/// production placement instead; every other failure stays fatal.
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

    await _verifyKnownOutput();
    await _verifyConvolutionalModel();
  });
}

Future<Uint8List> _loadAsset(String path) async {
  final data = await rootBundle.load(path);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

Future<void> _verifyKnownOutput() async {
  final model = CompiledModel.fromBuffer(
    await _loadAsset('assets/simple_model.tflite'),
    accelerators: const {Accelerator.gpu},
    precision: Precision.fp32,
  );
  try {
    expect(model.accelerators, const {Accelerator.gpu});
    final input = Float32List(model.inputByteSizes.single ~/ 4)..first = 3;
    final output = model.run([input]);
    expect(output, hasLength(1));
    expect(output.single.first, closeTo(7, 1e-3));
  } finally {
    model.close();
  }
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
    print(
      'Strict GPU MobileFaceNet: compile='
      '${compileWatch.elapsedMilliseconds}ms, inference='
      '${inferenceWatch.elapsedMilliseconds}ms',
    );
  } finally {
    model.close();
  }
}
