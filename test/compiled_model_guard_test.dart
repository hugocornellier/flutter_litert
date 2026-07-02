@TestOn('mac-os || linux || windows')
library;

import 'dart:typed_data';

import 'package:flutter_litert/src/compiled_model/compiled_model_native.dart';
import 'package:flutter_test/flutter_test.dart';

const _model = 'test/assets/add.tflite';

void main() {
  test('sync APIs throw while an async dispatch is in flight', () async {
    final cm = CompiledModel.fromFile(
      _model,
      tensorBufferMode: TensorBufferMode.hostMemory,
    );
    final input = Float32List(cm.inputByteSizes[0] ~/ 4);

    final pending = cm.runAsync([input]);
    // One microtask turn: the queued dispatch has written its inputs, set
    // the in-flight flag, and suspended awaiting the helper isolate. The
    // helper's reply needs an event-loop turn, so it cannot land before
    // these checks run.
    await null;

    expect(() => cm.run([input]), throwsStateError);
    expect(() => cm.writeInput(0, (i) {}), throwsStateError);
    expect(() => cm.dispatch(), throwsStateError);
    expect(() => cm.readOutput(0, (o) => o[0]), throwsStateError);
    expect(cm.close, throwsStateError);

    final out = await pending;
    expect(out, hasLength(cm.outputCount));

    // Fully usable and closable again once the dispatch completed.
    expect(cm.run([input]), hasLength(cm.outputCount));
    cm.close();
  });

  test('queued runAsync calls serialize and both complete', () async {
    final cm = CompiledModel.fromFile(
      _model,
      tensorBufferMode: TensorBufferMode.hostMemory,
    );
    final len = cm.inputByteSizes[0] ~/ 4;
    final a = Float32List(len)..fillRange(0, len, 1.0);
    final b = Float32List(len)..fillRange(0, len, 2.0);

    final results = await Future.wait([
      cm.runAsync([a]),
      cm.runAsync([b]),
    ]);
    // add.tflite computes x + x + x.
    expect(results[0][0][0], closeTo(3.0, 1e-6));
    expect(results[1][0][0], closeTo(6.0, 1e-6));
    cm.close();
  });
}
