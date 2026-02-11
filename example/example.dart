// ignore_for_file: avoid_print
import 'dart:io';

import 'package:flutter_litert/flutter_litert.dart';

/// Example demonstrating basic TFLite model inference.
///
/// To run this example, you need a TFLite model file.
/// See the full examples in the example/ subdirectories for complete
/// Flutter applications.
void main() async {
  // Load a model from a file
  final modelFile = File('/path/to/model.tflite');
  final interpreter = Interpreter.fromFile(modelFile);

  // Get input and output tensor info
  print('Input tensors: ${interpreter.getInputTensors()}');
  print('Output tensors: ${interpreter.getOutputTensors()}');

  // Prepare input data (shape depends on your model)
  var input = [
    [1.0, 2.0, 3.0, 4.0]
  ];

  // Prepare output buffer (shape depends on your model)
  var output = List.filled(1, List.filled(2, 0.0));

  // Run inference
  interpreter.run(input, output);

  print('Output: $output');

  // Clean up
  interpreter.close();
}
