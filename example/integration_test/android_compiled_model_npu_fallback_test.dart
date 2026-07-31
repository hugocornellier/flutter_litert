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

/// Verifies the default device group installed without an NPU vendor runtime.
///
/// This is deliberately suitable for a virtual Test Lab device: it proves a
/// strict NPU request fails clearly while a mixed NPU+CPU request removes the
/// unavailable NPU and remains usable on CPU. Vendor execution is covered by
/// `android_compiled_model_npu_test.dart` on physical hardware.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('unsupported Android NPU preserves explicit CPU fallback', (
    _,
  ) async {
    expect(Platform.isAndroid, isTrue);
    final device = await DeviceInfoPlugin().androidInfo;
    print(
      'Fallback device: ${device.manufacturer} ${device.model}; '
      'SDK ${device.version.sdkInt}; physical=${device.isPhysicalDevice}; '
      'hardware=${device.hardware}; ABIs=${device.supportedAbis.join(',')}',
    );

    final nativeLibraryDir = androidNativeLibraryDir;
    expect(nativeLibraryDir, isNotNull);
    final libraryNames = Directory(nativeLibraryDir!)
        .listSync()
        .whereType<File>()
        .map((file) => file.uri.pathSegments.last)
        .toSet();
    expect(
      libraryNames.where((name) => name.startsWith('libLiteRtDispatch_')),
      isEmpty,
      reason: 'The default device group must not install an NPU runtime.',
    );

    final bytes = await _loadAsset('assets/simple_model.tflite');
    expect(
      () => CompiledModel.fromBuffer(
        bytes,
        accelerators: const {Accelerator.npu},
      ),
      throwsA(isA<UnsupportedError>()),
      reason: 'A strict NPU request must not silently run on CPU.',
    );

    final mixed = CompiledModel.fromBuffer(
      bytes,
      accelerators: const {Accelerator.npu, Accelerator.cpu},
    );
    try {
      expect(mixed.accelerators, const {Accelerator.cpu});
      final input = Float32List(mixed.inputByteSizes.single ~/ 4)..first = 3;
      expect(mixed.run([input]).single.single, closeTo(7, 1e-5));
      print('Unavailable Android NPU cleanly fell back to CPU.');
    } finally {
      mixed.close();
    }
  });
}

Future<Uint8List> _loadAsset(String path) async {
  final data = await rootBundle.load(path);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
