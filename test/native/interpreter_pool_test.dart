import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/native.dart';

File get _modelFile =>
    File('${Directory.current.path}/test/assets/training_model.tflite');

void main() {
  test(
    'does not create isolate interpreters on macOS when delegates are disabled',
    () async {
      if (!Platform.isMacOS) {
        return;
      }

      final pool = InterpreterPool(poolSize: 1);
      await pool.initialize(
        (options, _) async =>
            Interpreter.fromFile(_modelFile, options: options),
        performanceConfig: PerformanceConfig.disabled,
      );

      try {
        await pool.withInterpreter((_, isolateInterpreter) async {
          expect(isolateInterpreter, isNull);
        });
      } finally {
        await pool.dispose();
      }
    },
  );
}
