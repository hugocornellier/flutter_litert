import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/src/native/tflite_status.dart';

void main() {
  test('known TfLiteStatus values retain number and name', () {
    expect(tfLiteStatusName(0), 'kTfLiteOk');
    expect(tfLiteStatusName(1), 'kTfLiteError');
    expect(tfLiteStatusName(2), 'kTfLiteDelegateError');
    expect(tfLiteStatusName(7), 'kTfLiteUnresolvedOps');
    expect(tfLiteStatusName(9), 'kTfLiteOutputShapeNotKnown');
    expect(describeTfLiteStatus(8), '8 (kTfLiteCancelled)');
  });

  test('unknown TfLiteStatus values remain diagnosable', () {
    expect(tfLiteStatusName(42), isNull);
    expect(describeTfLiteStatus(42), '42 (unrecognised TfLiteStatus)');
  });

  test('checkTfLiteStatus includes operation and symbolic status', () {
    expect(() => checkTfLiteStatus('invoke', 0), returnsNormally);
    expect(
      () => checkTfLiteStatus('TfLiteInterpreterInvoke', 2),
      throwsA(
        isA<StateError>().having(
          (error) => error.message,
          'message',
          contains(
            'TfLiteInterpreterInvoke failed with '
            'TfLiteStatus=2 (kTfLiteDelegateError)',
          ),
        ),
      ),
    );
  });
}
