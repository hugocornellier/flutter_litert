// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_litert/src/bindings/litert_loader.dart'
    show androidNativeLibraryDir;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Physical-device gate for Android LiteRT JIT NPU acceleration.
///
/// The default Test Lab invocation is deliberately a small, strict proof:
/// validate the packaged runtime, initialize CPU first, compile a known-output
/// graph with `{npu}` only, run it repeatedly, and recreate it. Set
/// `ANDROID_NPU_FULL_SWEEP=true` at build time to add representative face,
/// segmentation, and pose models with CPU-reference output comparisons.
///
/// There is no emulator skip. Android emulators have no vendor NPU and must
/// never be used for this gate.
const _runFullSweep = bool.fromEnvironment('ANDROID_NPU_FULL_SWEEP');
const _enforceFullSweepTolerance = bool.fromEnvironment(
  'ANDROID_NPU_ENFORCE_FULL_SWEEP_TOLERANCE',
  defaultValue: true,
);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'strict Qualcomm NPU compiles and runs on physical Android hardware',
    (tester) async {
      expect(
        Platform.isAndroid,
        isTrue,
        reason: 'This gate must only run on Android.',
      );

      final device = await DeviceInfoPlugin().androidInfo;
      print(
        'Test Lab device: ${device.manufacturer} ${device.model}; '
        'SDK ${device.version.sdkInt}; hardware=${device.hardware}; '
        'board=${device.board}; ABIs=${device.supportedAbis.join(',')}',
      );
      expect(
        device.isPhysicalDevice,
        isTrue,
        reason: 'A virtual device cannot validate a vendor NPU.',
      );
      expect(
        device.version.sdkInt,
        greaterThanOrEqualTo(31),
        reason: 'LiteRT Android NPU support requires API 31 or newer.',
      );
      expect(
        device.supportedAbis,
        contains('arm64-v8a'),
        reason: 'LiteRT Android NPU support is arm64-only.',
      );

      _verifyPackagedRuntime();
      await _verifyCpuThenStrictNpu();
      await _verifyRepeatedCreateRunClose();

      if (_runFullSweep) {
        await _verifyRepresentativeModels();
      } else {
        print(
          'Representative model sweep disabled for this smoke run. '
          'Dispatch with full_sweep=true to enable it.',
        );
      }
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );
}

Future<Uint8List> _loadAsset(String path) async {
  final data = await rootBundle.load(path);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

void _verifyPackagedRuntime() {
  final directoryPath = androidNativeLibraryDir;
  expect(
    directoryPath,
    isNotNull,
    reason:
        'The Android native-library directory was not extracted. '
        'useLegacyPackaging=true is required for Qualcomm NPU.',
  );
  final directory = Directory(directoryPath!);
  final names = directory
      .listSync()
      .whereType<File>()
      .map((file) => file.uri.pathSegments.last)
      .toSet();

  const commonLibraries = <String>{
    'libLiteRt.so',
    'libLiteRtCompilerPlugin_Qualcomm.so',
    'libLiteRtDispatch_Qualcomm.so',
    'libQnnHtp.so',
    'libQnnSystem.so',
    'libQnnHtpPrepare.so',
    'libQnnIr.so',
    'libQnnSaver.so',
  };
  for (final name in commonLibraries) {
    expect(
      names,
      contains(name),
      reason: '$name is missing from $directoryPath.',
    );
  }

  final skels = names
      .where(
        (name) =>
            RegExp(r'^libQnnHtpV(69|73|75|79|81)Skel[.]so$').hasMatch(name),
      )
      .toList();
  expect(
    skels,
    hasLength(1),
    reason: 'Exactly one Qualcomm HTP generation must be packaged: $skels.',
  );
  final version = RegExp(r'V(\d+)Skel').firstMatch(skels.single)!.group(1)!;
  expect(
    names,
    contains('libQnnHtpV${version}Stub.so'),
    reason: 'The HTP v$version stub must match ${skels.single}.',
  );
  print(
    'Qualcomm NPU runtime verified in $directoryPath '
    '(HTP v$version, ${commonLibraries.length + 2} required libraries).',
  );
}

Future<void> _verifyCpuThenStrictNpu() async {
  final bytes = await _loadAsset('assets/simple_model.tflite');

  // This catches accidentally reusing the ordinary environment, which lacks
  // the compiler/dispatch directory options if CPU was initialized first.
  final cpu = CompiledModel.fromBuffer(
    bytes,
    accelerators: const {Accelerator.cpu},
    precision: Precision.fp32,
  );
  try {
    final input = Float32List(cpu.inputByteSizes.single ~/ 4)..first = 3;
    expect(cpu.run([input]).single.single, closeTo(7, 1e-5));
  } finally {
    cpu.close();
  }

  final compileWatch = Stopwatch()..start();
  final npu = CompiledModel.fromBuffer(
    bytes,
    accelerators: const {Accelerator.npu},
    precision: Precision.fp32,
  );
  compileWatch.stop();
  try {
    expect(npu.accelerators, const {Accelerator.npu});
    expect(
      npu.isFullyAccelerated,
      isTrue,
      reason: 'The strict known-output graph was not fully NPU-accelerated.',
    );

    final input = Float32List(npu.inputByteSizes.single ~/ 4);
    final inferenceTimes = <int>[];
    for (var i = 0; i < 5; i++) {
      input.first = i + 1;
      final inferenceWatch = Stopwatch()..start();
      final result = npu.run([input]).single.single;
      inferenceWatch.stop();
      inferenceTimes.add(inferenceWatch.elapsedMicroseconds);
      expect(result, closeTo(2 * (i + 1) + 1, 1e-4));
    }
    print(
      'Strict NPU known-output model verified: '
      'compile=${compileWatch.elapsedMilliseconds}ms, '
      'inference_us=$inferenceTimes.',
    );
  } finally {
    npu.close();
  }
}

Future<void> _verifyRepeatedCreateRunClose() async {
  final bytes = await _loadAsset('assets/simple_model.tflite');
  for (var i = 0; i < 3; i++) {
    final model = CompiledModel.fromBuffer(
      bytes,
      accelerators: const {Accelerator.npu},
      precision: Precision.fp32,
    );
    try {
      final input = Float32List(model.inputByteSizes.single ~/ 4)
        ..first = i + 2;
      expect(model.run([input]).single.single, closeTo(2 * (i + 2) + 1, 1e-4));
      expect(model.isFullyAccelerated, isTrue);
    } finally {
      model.close();
    }
  }
  print('Three strict NPU create/run/close cycles verified.');
}

Future<void> _verifyRepresentativeModels() async {
  const models = <String>[
    'assets/mobilefacenet.tflite',
    'assets/selfie_multiclass.tflite',
    'assets/pose_landmark_heavy.tflite',
  ];
  final executionFailures = <String>[];
  final accuracyRejections = <String>[];

  for (final path in models) {
    final bytes = await _loadAsset(path);
    final compileWatch = Stopwatch()..start();
    CompiledModel? model;
    try {
      model = CompiledModel.fromBuffer(
        bytes,
        accelerators: const {Accelerator.npu, Accelerator.cpu},
        precision: Precision.fp32,
      );
      compileWatch.stop();
      final verifyWatch = Stopwatch()..start();
      final verification = verifyCompiledModel(bytes, model);
      verifyWatch.stop();
      print(
        '$path NPU+CPU verified: compile=${compileWatch.elapsedMilliseconds}ms, '
        'verification=${verifyWatch.elapsedMilliseconds}ms, $verification.',
      );

      if (verification.skipped || verification.absoluteDeviation <= 0) {
        executionFailures.add('$path: $verification');
      } else if (!verification.agrees) {
        accuracyRejections.add('$path: $verification');
      }
    } catch (error) {
      compileWatch.stop();
      print(
        '$path NPU+CPU failed after '
        '${compileWatch.elapsedMilliseconds}ms: $error',
      );
      executionFailures.add('$path: threw $error');
    } finally {
      model?.close();
    }
  }

  expect(
    executionFailures,
    isEmpty,
    reason:
        'Every representative model must compile, run, and demonstrate a '
        'nonzero NPU-vs-CPU deviation:\n${executionFailures.join('\n')}',
  );

  if (accuracyRejections.isEmpty) {
    print('All representative models passed the CPU-reference tolerance.');
    return;
  }

  print(
    'Representative models rejected by the CPU-reference tolerance:\n'
    '${accuracyRejections.join('\n')}',
  );
  if (!_enforceFullSweepTolerance) {
    print(
      'Diagnostic sweep recorded the rejections without failing the physical '
      'runtime matrix.',
    );
    return;
  }

  expect(
    accuracyRejections,
    isEmpty,
    reason:
        'Models outside the 1% CPU-reference tolerance must not be enabled '
        'without application-level accuracy validation:\n'
        '${accuracyRejections.join('\n')}',
  );
}
