import 'dart:io';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // Each mode must give usable options on every host, whether or not its
  // delegate exists there. PerformanceConfig.gpu() in particular is passed by
  // apps without a platform check, so it must never throw.
  for (final mode in PerformanceMode.values) {
    test('InterpreterFactory.create($mode) runs the model', () {
      final (options, delegate) = InterpreterFactory.create(
        PerformanceConfig(mode: mode),
      );
      final interpreter = Interpreter.fromFile(
        File('test/assets/add.tflite'),
        options: options,
      );
      addTearDown(() {
        interpreter.close();
        options.delete();
        delegate?.delete();
      });

      interpreter.allocateTensors();
      interpreter.invoke();
      expect(interpreter.getOutputTensors(), isNotEmpty);
    });
  }

  test('PerformanceConfig.gpu() uses no GPU delegate on Linux or Windows', () {
    if (!Platform.isLinux && !Platform.isWindows) {
      markTestSkipped('only Linux and Windows have no GPU mapping');
      return;
    }
    final (options, delegate) = InterpreterFactory.create(
      const PerformanceConfig.gpu(),
    );
    addTearDown(options.delete);
    expect(delegate, isNull);
  });
}
