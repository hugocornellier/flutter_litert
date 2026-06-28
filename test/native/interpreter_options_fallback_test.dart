import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('CPU fallback options preserve threads and omit delegates', () {
    final delegateOptions = XNNPackDelegateOptions(numThreads: 3);
    final delegate = XNNPackDelegate(options: delegateOptions);
    final options = InterpreterOptions()
      ..threads = 3
      ..addDelegate(delegate);

    final fallback = options.copyWithoutDelegates();
    addTearDown(() {
      fallback.delete();
      options.delete();
      delegate.delete();
      delegateOptions.delete();
    });

    expect(options.hasDelegate, isTrue);
    expect(fallback.hasDelegate, isFalse);
    expect(fallback.threads, 3);
  });
}
