import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_litert/src/bindings/litert_loader.dart'
    show litertRuntimeDir, macOsCoreMlNpuAcceleratorPath;

/// End-to-end CompiledModel (LiteRT Next) checks inside a real app process.
///
/// CPU tests must pass wherever the LiteRT runtime is bundled, across all five
/// native platforms. On desktop this proves the production load path: the
/// platform build bundles the runtime next to the executable and the loader
/// resolves it from there, not from the package checkout. GPU tests assert
/// success when the platform accelerator initializes and skip when it cannot
/// (headless or emulated environments without a working GPU stack, or
/// armeabi-v7a Android, the one ABI upstream ships no accelerator for).
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late File modelFile;
  late Uint8List modelBytes;

  late Uint8List segModelBytes;

  setUpAll(() async {
    final data = await rootBundle.load('assets/simple_model.tflite');
    modelBytes = data.buffer.asUint8List();
    final tmpDir = await Directory.systemTemp.createTemp('litert_cm_test_');
    modelFile = File('${tmpDir.path}/simple_model.tflite');
    await modelFile.writeAsBytes(modelBytes);

    final segData = await rootBundle.load('assets/selfie_multiclass.tflite');
    segModelBytes = segData.buffer.asUint8List();
  });

  group('CompiledModel (LiteRT Next)', () {
    testWidgets('loads the runtime from the app bundle on desktop', (
      tester,
    ) async {
      if (!(Platform.isMacOS || Platform.isLinux || Platform.isWindows)) {
        markTestSkipped('Bundle-origin check is for desktop bundling.');
        return;
      }
      // Force the lazy loader, then assert the library it actually dlopen'd
      // came from inside the built app, and not from the loader's
      // package-checkout fallback, which exists in CI/dev environments but not
      // in shipped apps. This is what makes a green run prove the bundling
      // worked.
      CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      ).close();
      final execDir = File(Platform.resolvedExecutable).parent;

      if (Platform.isMacOS) {
        // CocoaPods and SwiftPM both ship libLiteRt.dylib as a *resource*, so
        // unlike the CMake desktops its directory depends on which resource
        // layout the toolchain picked (the candidates in delegateBundlePaths).
        // Assert containment in the .app rather than one fixed path; what must
        // not happen is resolving to the package checkout outside it.
        final contentsDir = execDir.parent.resolveSymbolicLinksSync();
        expect(litertRuntimeDir, isNotNull);
        expect(
          litertRuntimeDir,
          startsWith(contentsDir),
          reason:
              'runtime resolved outside the app bundle, so the podspec '
              'resource bundling did not take effect',
        );
        return;
      }

      // CMake bundled_libraries puts the runtime next to the executable.
      final expectedDir = Platform.isLinux
          ? '${execDir.path}/lib'
          : execDir.path;
      expect(
        litertRuntimeDir,
        Directory(expectedDir).resolveSymbolicLinksSync(),
      );
    });

    testWidgets('compiles and runs on the CPU accelerator', (tester) async {
      final cm = CompiledModel.fromFile(
        modelFile.path,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(cm.close);

      expect(cm.accelerators, {Accelerator.cpu});
      expect(cm.inputCount, 1);
      expect(cm.outputCount, 1);

      // Model is y = 2*x + 1.
      final input = Float32List(cm.inputByteSizes[0] ~/ 4);
      input[0] = 3.0;
      final output = cm.run([input]);
      expect(output[0][0], closeTo(7.0, 1e-3));
    });

    testWidgets('runs strict NPU from an Apple app bundle', (tester) async {
      if (!(Platform.isIOS || Platform.isMacOS)) {
        markTestSkipped('The Core ML NPU accelerator is Apple-platform only.');
        return;
      }
      if (Platform.isMacOS && Abi.current() != Abi.macosArm64) {
        markTestSkipped('The Core ML NPU accelerator requires Apple Silicon.');
        return;
      }

      final cm = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.npu},
      );
      addTearDown(cm.close);

      if (Platform.isMacOS) {
        final bridgePath = macOsCoreMlNpuAcceleratorPath;
        expect(bridgePath, isNotNull);
        final loadedBridgePath = bridgePath!;
        final contentsDir = File(
          Platform.resolvedExecutable,
        ).parent.parent.resolveSymbolicLinksSync();
        expect(
          File(loadedBridgePath).resolveSymbolicLinksSync(),
          startsWith(contentsDir),
          reason:
              'NPU bridge resolved outside the built app, so this run did not '
              'prove resource bundling',
        );
        expect(
          File(
            '${File(loadedBridgePath).parent.path}/'
            'libtensorflowlite_coreml_npu-mac.dylib',
          ).existsSync(),
          isTrue,
          reason: 'the NPU bridge and its dedicated delegate must be siblings',
        );
      }

      final input = Float32List(cm.inputByteSizes.first ~/ 4)..first = 3;
      expect(cm.run([input]).single.single, closeTo(7, 1e-3));
      expect(cm.isFullyAccelerated, isTrue);
    });

    testWidgets('compiles from bytes and runAsync matches run', (tester) async {
      final cm = CompiledModel.fromBuffer(
        modelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(cm.close);

      final input = Float32List(cm.inputByteSizes[0] ~/ 4);
      input[0] = -1.0;
      final syncOut = cm.run([input]);
      final asyncOut = await cm.runAsync([input]);
      expect(asyncOut, syncOut);
      expect(syncOut[0][0], closeTo(-1.0, 1e-3));
    });

    // Every other CPU test here runs simple_model, whose single-float output
    // makes the output-buffer lock trivial. face_detection_tflite hits an
    // intermittent 'LiteRtLockTensorBuffer output[0] failed with
    // LiteRtStatus=3' on Windows when reading this model's much larger
    // output, a path nothing in this package covered. Repeats the run so a
    // single successful lock is not mistaken for a stable one.
    testWidgets('locks and reads a large multi-class segmentation output', (
      tester,
    ) async {
      final cm = CompiledModel.fromBuffer(
        segModelBytes,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(cm.close);

      final outBytes = cm.outputByteSizes[0];
      expect(
        outBytes,
        greaterThan(1 << 20),
        reason:
            'the point of this test is a large output buffer; a small '
            'one would not exercise the failing path',
      );

      final input = Float32List(cm.inputByteSizes[0] ~/ 4);
      for (var i = 0; i < 5; i++) {
        final output = cm.run([input]);
        expect(output[0], hasLength(outBytes ~/ 4), reason: 'iteration $i');
        expect(
          output[0].every((v) => v.isFinite),
          isTrue,
          reason: 'iteration $i produced non-finite values',
        );
      }
    });

    testWidgets('host-memory buffers match managed buffers on CPU', (
      tester,
    ) async {
      final managed = CompiledModel.fromFile(
        modelFile.path,
        accelerators: {Accelerator.cpu},
      );
      final hostMemory = CompiledModel.fromFile(
        modelFile.path,
        accelerators: {Accelerator.cpu},
        tensorBufferMode: TensorBufferMode.hostMemory,
      );
      addTearDown(managed.close);
      addTearDown(hostMemory.close);

      final input = Float32List(managed.inputByteSizes[0] ~/ 4);
      input[0] = 5.0;
      expect(hostMemory.run([input]), managed.run([input]));
    });

    testWidgets('compiles with GPU and CPU fallback', (tester) async {
      final CompiledModel cm;
      try {
        cm = CompiledModel.fromFile(
          modelFile.path,
          accelerators: {Accelerator.gpu, Accelerator.cpu},
        );
      } on StateError catch (e) {
        // A bundled GPU accelerator that can't initialize (e.g. Linux WebGPU
        // with no adapter on a headless runner) fails the create outright
        // instead of falling back to CPU, so skip where the GPU stack is
        // absent; the CPU-fallback path is still covered by the CPU tests.
        if (_isGpuUnavailable(e)) {
          markTestSkipped('GPU accelerator unavailable here.');
          return;
        }
        rethrow;
      }
      addTearDown(cm.close);

      // Sync run on purpose. On the iOS *simulator*, MTLSimDriver's shared
      // events can wedge when an async-run Metal model is closed and a new
      // environment then waits on an async event (reproduced reliably with
      // async here followed by the async strict-GPU test below; sync-then-
      // async sequences are fine). Real devices use a different Metal driver.
      final input = Float32List(cm.inputByteSizes[0] ~/ 4);
      input[0] = 3.0;
      final output = cm.run([input]);
      expect(output[0][0], closeTo(7.0, 1e-3));
    });

    testWidgets('GPU fallback factory always yields a working model', (
      tester,
    ) async {
      Object? fallbackError;
      final cm = await CompiledModel.fromBufferWithGpuFallbackAsync(
        modelBytes,
        onFallback: (error) => fallbackError = error,
      );
      addTearDown(cm.close);

      if (fallbackError != null) {
        expect(cm.accelerators, {Accelerator.cpu});
      }
      final input = Float32List(cm.inputByteSizes[0] ~/ 4);
      input[0] = 3.0;
      final output = cm.run([input]);
      expect(output, hasLength(cm.outputCount));
      expect(output[0], hasLength(cm.outputByteSizes[0] ~/ 4));
      expect(output[0][0], closeTo(7.0, 1e-3));
    });

    testWidgets('compiles and runs on the strict GPU accelerator '
        'when the GPU stack is available', (tester) async {
      final CompiledModel cm;
      try {
        cm = CompiledModel.fromFile(
          modelFile.path,
          accelerators: {Accelerator.gpu},
        );
      } on StateError catch (e) {
        if (_isGpuUnavailable(e)) {
          markTestSkipped('GPU accelerator unavailable here.');
          return;
        }
        rethrow;
      }
      addTearDown(cm.close);

      final input = Float32List(cm.inputByteSizes[0] ~/ 4);
      input[0] = 3.0;
      final output = await cm.runAsync([input]);
      expect(output[0][0], closeTo(7.0, 1e-3));
    });
  });
}

/// Whether [e] means "the GPU accelerator can't run here", as opposed to a
/// real failure.
///
/// Every native platform bundles a GPU accelerator: Metal ships with the
/// package on Apple platforms, Windows and Linux download the WebGPU
/// (Dawn) accelerator at build time in the platform CMakeLists, and Android
/// extracts the OpenCL/GL accelerator from the LiteRT AAR for arm64-v8a and
/// x86_64 (upstream ships none for armeabi-v7a). A skip here is therefore
/// about the machine, never missing packaging. On Apple the accelerator
/// always initializes, so only the documented gpu-only compilation refusal
/// (status 504, an op with no GPU implementation) is acceptable. Elsewhere
/// the stack itself can be absent: Android emulators have no working OpenCL,
/// and headless Windows/Linux runners expose no usable WebGPU adapter, so on
/// those platforms any LiteRT status from a strict-GPU create means the
/// accelerator cannot run here. Real-hardware GPU execution is validated
/// separately by the Firebase Test Lab gate
/// (android_compiled_model_gpu_test.dart).
bool _isGpuUnavailable(StateError e) {
  if (e.message.contains('LiteRtStatus=504')) return true;
  return (Platform.isAndroid || Platform.isLinux || Platform.isWindows) &&
      e.message.contains('LiteRtStatus=');
}
