import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

/// Collects what [InterpreterFactory] writes to stderr.
class _CapturedStderr implements Stdout {
  final lines = <String>[];

  @override
  void writeln([Object? object = '']) => lines.add('$object');

  @override
  void write(Object? object) => lines.add('$object');

  @override
  dynamic noSuchMethod(Invocation invocation) => null;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List bytes;

  setUpAll(() async {
    bytes = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();
  });

  // Regression coverage for classic Interpreter creation with auto mode on an
  // iOS simulator. Some models cannot be applied to the Metal delegate and
  // must retry on CPU; others can run through Metal directly.
  testWidgets(
    'Interpreter falls back to CPU when the auto delegate cannot initialize',
    (tester) async {
      // auto -> GPU/Metal on iOS. Creation must succeed whether this model is
      // delegated or requires the CPU fallback.
      final (options, delegate) = InterpreterFactory.create(
        const PerformanceConfig(),
      );
      final interpreter = Interpreter.fromBuffer(bytes, options: options);
      addTearDown(() {
        interpreter.close();
        options.delete();
        delegate?.delete();
      });

      // Reaching here means creation did not throw; the effective backend is
      // also exposed so callers can make post-creation scheduling decisions.
      expect(interpreter, isA<Interpreter>());
      expect(interpreter.getInputTensors(), isNotEmpty);
      expect(interpreter.hasActiveDelegate, isA<bool>());
    },
  );

  // PerformanceConfig.gpu() is public API that apps and detector packages pass
  // without checking the platform. Since 4.0.0 Android has no Interpreter GPU
  // delegate, and the factory must still hand back a working interpreter.
  testWidgets('PerformanceConfig.gpu never throws and runs the model', (
    tester,
  ) async {
    final captured = _CapturedStderr();
    final results = IOOverrides.runZoned(() {
      return [
        for (var i = 0; i < 3; i++)
          InterpreterFactory.create(const PerformanceConfig.gpu()),
      ];
    }, stderr: () => captured);

    for (final (options, delegate) in results) {
      final interpreter = Interpreter.fromBuffer(bytes, options: options);
      addTearDown(() {
        interpreter.close();
        options.delete();
        delegate?.delete();
      });
      interpreter.getInputTensor(0).asFloat32View()[0] = 3.0;
      interpreter.invoke();
      expect(
        interpreter.getOutputTensor(0).asFloat32View()[0],
        closeTo(7.0, 1e-3),
      );

      if (Platform.isAndroid && androidRuntime == LiteRtRuntime.bundled) {
        // The bundled fallback is XNNPACK, not bare CPU, which is often
        // several times slower.
        expect(delegate, isA<XNNPackDelegate>());
        expect(interpreter.hasActiveDelegate, isTrue);
      }
    }

    if (Platform.isAndroid && androidRuntime == LiteRtRuntime.bundled) {
      final warnings = captured.lines
          .where((line) => line.contains('PerformanceConfig.gpu()'))
          .toList();
      expect(warnings, hasLength(1), reason: 'warned once for three calls');
      expect(warnings.single, contains('removed in flutter_litert 4.0.0'));
      expect(warnings.single, contains('CompiledModel with Accelerator.gpu'));
    }
  });
}
