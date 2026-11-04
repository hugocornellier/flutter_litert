import 'dart:io';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert/src/bindings/android_runtime_shim.dart';
import 'package:flutter_litert/src/bindings/litert_ffi.dart';
import 'package:flutter_test/flutter_test.dart';

/// Host-side coverage for flutterLitert.runtime. The Play services runtime
/// itself only exists on Android; example/play_services_test_host runs it.
void main() {
  group('off Android', () {
    test('the runtime is bundled and needs no initialization', () async {
      expect(androidRuntime, LiteRtRuntime.bundled);
      expect(usesPlayServicesRuntime, isFalse);
      expect(androidRuntimeShim, isNull);

      final status = await initializeAndroidRuntime(enableGpu: true);
      expect(status.runtime, LiteRtRuntime.bundled);
      expect(status.isReady, isTrue);
      expect(status.gpuAvailable, isNull);
    });
  });

  group('LiteRtRuntimeStatus', () {
    test('bundled is ready with nothing to report', () {
      const status = LiteRtRuntimeStatus.bundled();
      expect(status.interpreterReady, isTrue);
      expect(status.compiledModelReady, isTrue);
      expect(status.isReady, isTrue);
      expect(status.interpreterError, isNull);
      expect(status.compiledModelError, isNull);
      expect(status.gpuError, isNull);
      expect(status.playServicesError, isNull);
      expect(
        status.toString(),
        'LiteRtRuntimeStatus(runtime: bundled, interpreterReady: true, '
        'compiledModelReady: true, gpuAvailable: null)',
      );
    });

    test('an auto fallback to bundled says why', () {
      const status = LiteRtRuntimeStatus.bundled(
        playServicesError: 'Google Play services could not provide LiteRT',
      );
      expect(status.runtime, LiteRtRuntime.bundled);
      expect(status.isReady, isTrue);
      expect(
        status.toString(),
        endsWith(
          'playServicesError: Google Play services could not provide LiteRT)',
        ),
      );
    });

    test('is not ready when either half is not', () {
      const partial = LiteRtRuntimeStatus(
        runtime: LiteRtRuntime.playServices,
        interpreterReady: false,
        compiledModelReady: true,
        gpuAvailable: false,
        interpreterError: 'TfLiteNative.initialize failed: x',
        gpuError: 'LiteRt.initializeNative with GPU failed: y',
      );
      expect(partial.isReady, isFalse);
      expect(partial.toString(), contains('interpreterError: TfLiteNative'));
      expect(partial.toString(), contains('gpuError: LiteRt'));
      expect(partial.toString(), isNot(contains('compiledModelError')));
    });

    test('the unavailable exception names both causes', () {
      const exception = LiteRtRuntimeUnavailableException(
        LiteRtRuntimeStatus(
          runtime: LiteRtRuntime.playServices,
          interpreterReady: false,
          compiledModelReady: false,
          interpreterError: 'no Play services',
          compiledModelError: 'download not allowed',
        ),
      );
      expect(exception.toString(), contains('Interpreter: no Play services'));
      expect(
        exception.toString(),
        contains('CompiledModel: download not allowed'),
      );
    });
  });

  group('checkPlayServicesRuntimeReady', () {
    test('passes once the runtime is ready', () {
      checkPlayServicesRuntimeReady(
        component: 'Interpreter',
        state: kRuntimeStateReady,
        error: () => fail('the error is only read on failure'),
      );
    });

    test('asks for initialization before it was attempted', () {
      expect(
        () => checkPlayServicesRuntimeReady(
          component: 'CompiledModel',
          state: kRuntimeStateNotInitialized,
          error: () => '',
        ),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('flutterLitert.runtime=playServices'),
              contains('await initializeAndroidRuntime()'),
              contains('CompiledModel'),
            ),
          ),
        ),
      );
    });

    test('reports why initialization failed', () {
      expect(
        () => checkPlayServicesRuntimeReady(
          component: 'Interpreter',
          state: kRuntimeStateFailed,
          error: () => 'TfLiteNative.initialize failed: no module',
        ),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('TfLiteNative.initialize failed: no module'),
              contains('initializeAndroidRuntime() again'),
            ),
          ),
        ),
      );
    });
  });

  group('gpuError', () {
    String? errorFor({
      bool enableGpu = true,
      bool compiledModelReady = true,
      bool hasGpuAccelerator = false,
      String? reason,
      String? pluginGpuError,
    }) => gpuErrorForCall(
      enableGpu: enableGpu,
      compiledModelReady: compiledModelReady,
      hasGpuAccelerator: hasGpuAccelerator,
      reason: reason,
      pluginGpuError: pluginGpuError,
    );

    test('is null for a call that did not ask for the GPU', () {
      expect(errorFor(enableGpu: false, reason: 'r'), isNull);
    });

    test('is null when the GPU accelerator is there', () {
      expect(errorFor(hasGpuAccelerator: true, reason: 'r'), isNull);
    });

    test('reports the reason recorded when the runtime was attached', () {
      expect(errorFor(reason: 'r', pluginGpuError: 'later'), 'r');
    });

    test('reports this call\'s GPU failure while the runtime is not ready', () {
      expect(
        errorFor(compiledModelReady: false, reason: 'r', pluginGpuError: 'p'),
        'p',
      );
    });

    test('falls back to a generic reason when none was recorded', () {
      expect(errorFor(), 'the CompiledModel runtime has no GPU accelerator');
    });
  });

  test('unsupported features name the Play services runtime', () {
    final error = playServicesUnsupported('Accelerator.npu', 'Use the GPU.');
    expect(
      error.message,
      'flutter_litert: Accelerator.npu is not available with the Google Play '
      'services LiteRT runtime (flutterLitert.runtime=playServices). Use the '
      'GPU.',
    );
  });

  group('Play services forwarders', () {
    // litert_ffi.dart binds LiteRt functions by name. In Play services mode
    // those names resolve in libflutter_litert_runtime.so, which exports a
    // forwarder for each one listed in litert_forwarders.inc. A binding with no
    // forwarder would fail at load time on Android only, so pin the two lists
    // together here, where every CI job runs.
    Set<String> matches(String path, RegExp pattern) => pattern
        .allMatches(File(path).readAsStringSync())
        .map((m) => m.group(1)!)
        .toSet();

    test('cover exactly the LiteRt functions the Dart bindings look up', () {
      final bound = matches(
        'lib/src/bindings/litert_ffi.dart',
        RegExp(r">\(\s*'(LiteRt\w+)'\s*\)"),
      );
      final forwarded = matches(
        'src/play_services/litert_forwarders.inc',
        RegExp(r'FLUTTER_LITERT_FORWARD_(?:STATUS|VOID)\((LiteRt\w+),'),
      );
      expect(bound, isNotEmpty);
      expect(
        forwarded,
        bound,
        reason:
            'Re-run scripts/generate_play_services_forwarders.py after '
            'changing the LiteRt functions litert_ffi.dart binds.',
      );
    });

    test('the C shim pins the env option values litert_ffi.dart uses', () {
      final shim = File(
        'src/play_services/flutter_litert_runtime.c',
      ).readAsStringSync();
      expect(shim, contains('kLiteRtAnyTypeVoidPtr == $kLiteRtAnyTypeVoidPtr'));
      expect(
        shim,
        contains('kLiteRtEnvOptionTagContext == $kLiteRtEnvOptionTagContext'),
      );
      expect(
        shim,
        contains(
          'kLiteRtEnvOptionTagSystemRuntimeHandle == '
          '$kLiteRtEnvOptionTagSystemRuntimeHandle',
        ),
      );
      expect(
        shim,
        contains(
          'kLiteRtEnvOptionTagSystemGpuAcceleratorHandle == '
          '$kLiteRtEnvOptionTagSystemGpuAcceleratorHandle',
        ),
      );
    });
  });
}
