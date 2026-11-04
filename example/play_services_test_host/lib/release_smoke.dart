import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_litert/native.dart';

/// Release build smoke test for the Play services runtime.
///
/// Release builds shrink the plugin's Java and Kotlin with R8 and leave out
/// package:integration_test, so CI builds this entry point with
/// `flutter build apk --release -t lib/release_smoke.dart`, launches it, and
/// reads the result line from logcat:
///
///   flutter_litert release smoke: PASS ...
///   flutter_litert release smoke: FAIL ...
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  String result;
  try {
    final status = await initializeAndroidRuntime(enableGpu: true);
    final bytes = (await rootBundle.load(
      'assets/simple_model.tflite',
    )).buffer.asUint8List();

    final interpreter = Interpreter.fromBuffer(bytes);
    interpreter.getInputTensor(0).asFloat32View()[0] = 3.0;
    interpreter.invoke();
    final y1 = interpreter.getOutputTensor(0).asFloat32View()[0];
    interpreter.close();

    final model = CompiledModel.fromBuffer(bytes);
    final input = Float32List(model.inputByteSizes[0] ~/ 4)..[0] = 3.0;
    final y2 = model.run([input])[0][0];
    model.close();

    // simple_model is y = 2*x + 1.
    final ok = (y1 - 7.0).abs() < 1e-3 && (y2 - 7.0).abs() < 1e-3;
    result =
        '${ok ? 'PASS' : 'FAIL'} interpreter=$y1 compiledModel=$y2 '
        'status=$status';
  } catch (error) {
    result = 'FAIL $error';
  }
  // ignore: avoid_print
  print('flutter_litert release smoke: $result');
  runApp(const SizedBox.shrink());
}
