@TestOn('mac-os || linux || windows')
library;

import 'dart:io';

import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// A failed [InterpreterPool.initialize] must leave nothing behind.
///
/// The regression this guards: `initialize` only disposes first when
/// `_isInitialized` is true, and a failed call never sets it. So before this was
/// fixed, retrying after a partial failure accumulated the earlier attempt's
/// interpreters (a pool of 3 ended up holding 4), each one alive with native
/// memory and an XNNPACK threadpool but absent from the round-robin pool and
/// therefore never used.
void main() {
  const model = 'example/assets/simple_model.tflite';

  Future<void> initFailingAt(InterpreterPool pool, int failingCall) async {
    var calls = 0;
    await pool.initialize((options, delegate) async {
      calls++;
      if (calls == failingCall) throw StateError('slot $failingCall failed');
      return Interpreter.fromFile(File(model), options: options);
    });
  }

  test('a partially-failed initialize releases what it built', () async {
    final pool = InterpreterPool(poolSize: 3);
    addTearDown(pool.dispose);

    await expectLater(initFailingAt(pool, 2), throwsA(isA<StateError>()));

    expect(pool.isInitialized, isFalse);
    expect(
      pool.interpreters,
      isEmpty,
      reason: 'the successful slots should have been released',
    );
  });

  test('failing on the very first slot also leaves nothing behind', () async {
    final pool = InterpreterPool(poolSize: 3);
    addTearDown(pool.dispose);

    await expectLater(initFailingAt(pool, 1), throwsA(isA<StateError>()));

    expect(pool.isInitialized, isFalse);
    expect(pool.interpreters, isEmpty);
  });

  test(
    'retrying after a failure yields exactly poolSize interpreters',
    () async {
      final pool = InterpreterPool(poolSize: 3);
      addTearDown(pool.dispose);

      await expectLater(initFailingAt(pool, 2), throwsA(isA<StateError>()));

      // The natural recovery. This is the case that used to over-allocate.
      await pool.initialize((options, delegate) async {
        return Interpreter.fromFile(File(model), options: options);
      });

      expect(pool.isInitialized, isTrue);
      expect(pool.interpreters, hasLength(3));
    },
  );

  test(
    'the pool is usable after a failed-then-successful initialize',
    () async {
      final pool = InterpreterPool(poolSize: 2);
      addTearDown(pool.dispose);

      await expectLater(initFailingAt(pool, 2), throwsA(isA<StateError>()));
      await pool.initialize((options, delegate) async {
        return Interpreter.fromFile(File(model), options: options);
      });

      // Proves the surviving slots are live, not just correctly counted.
      final ran = await pool.withInterpreter((interpreter, isolate) async {
        return interpreter.getInputTensors().isNotEmpty;
      });
      expect(ran, isTrue);
    },
  );
}
