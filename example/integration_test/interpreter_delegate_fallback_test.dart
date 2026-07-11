import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Regression coverage for classic Interpreter creation with auto mode on an
  // iOS simulator. Some models cannot be applied to the Metal delegate and
  // must retry on CPU; others can run through Metal directly.
  testWidgets(
    'Interpreter falls back to CPU when the auto delegate cannot initialize',
    (tester) async {
      final bytes = (await rootBundle.load(
        'assets/simple_model.tflite',
      )).buffer.asUint8List();

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
}
