// ignore_for_file: avoid_print
import 'dart:io';

import 'package:flutter_litert/native.dart';

/// Minimal native file-based inference example.
///
/// Run from the example directory so the bundled model path resolves:
/// `dart run example.dart`
void main() async {
  final modelFile = File('assets/simple_model.tflite');
  final interpreter = Interpreter.fromFile(modelFile);

  print('Input tensors: ${interpreter.getInputTensors()}');
  print('Output tensors: ${interpreter.getOutputTensors()}');

  var input = [
    [3.0],
  ];

  var output = [
    [0.0],
  ];

  interpreter.run(input, output);

  print('Output: $output');

  interpreter.close();
}
