import 'dart:typed_data';

import 'package:flutter_litert/src/compiled_model/compiled_model_unsupported.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final bytes = Uint8List.fromList([1]);

  test('policy factories preserve the unsupported-platform surface', () {
    expect(
      () => CompiledModel.fromFileWithConfig('model.tflite'),
      throwsUnsupportedError,
    );
    expect(
      () => CompiledModel.fromBufferWithConfig(bytes),
      throwsUnsupportedError,
    );
    expect(
      () => CompiledModel.fromBufferWithConfigAsync(bytes),
      throwsUnsupportedError,
    );
  });
}
