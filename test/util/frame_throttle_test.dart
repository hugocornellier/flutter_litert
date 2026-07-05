import 'dart:async';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FrameThrottle', () {
    test('runs the task and returns its result when idle', () async {
      final t = FrameThrottle();
      final result = await t.run(() async => 42);
      expect(result, 42);
      expect(t.isBusy, isFalse);
    });

    test('drops the task (returns null) while one is in flight', () async {
      final t = FrameThrottle();
      final gate = Completer<void>();

      final first = t.run(() async {
        await gate.future;
        return 'first';
      });
      expect(t.isBusy, isTrue);

      // A second frame arrives while the first is still being processed.
      final second = await t.run(() async => 'second');
      expect(second, isNull);

      gate.complete();
      expect(await first, 'first');
      expect(t.isBusy, isFalse);
    });

    test('clears the busy flag after the task throws', () async {
      final t = FrameThrottle();
      await expectLater(
        t.run<int>(() async => throw StateError('boom')),
        throwsStateError,
      );
      expect(t.isBusy, isFalse);
      // A subsequent run still executes.
      expect(await t.run(() async => 7), 7);
    });

    test('runs again after a previous task completes', () async {
      final t = FrameThrottle();
      expect(await t.run(() async => 1), 1);
      expect(await t.run(() async => 2), 2);
    });

    test('reset() clears the busy flag', () async {
      final t = FrameThrottle();
      final gate = Completer<void>();
      final inFlight = t.run(() async {
        await gate.future;
        return 0;
      });
      expect(t.isBusy, isTrue);
      t.reset();
      expect(t.isBusy, isFalse);
      // Let the in-flight task finish so no future is left dangling.
      gate.complete();
      await inFlight;
    });
  });
}
