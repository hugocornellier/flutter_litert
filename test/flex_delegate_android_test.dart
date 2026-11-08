@TestOn('vm')
library;

import 'package:flutter_litert/src/delegates/flex_delegate_native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Android flex was removed in 4.0.0. The real Android path only runs on a
/// device, so these tests make the host look like Android.
void main() {
  group('FlexDelegate on Android', () {
    setUp(() => FlexDelegate.debugIsAndroidOverride = true);
    tearDown(() => FlexDelegate.debugIsAndroidOverride = null);

    final removed = isA<UnsupportedError>().having(
      (e) => e.message,
      'message',
      allOf(
        contains('removed in flutter_litert 4.0.0'),
        contains('LiteRT 2.x has no flex delegate'),
        contains('Reconvert the model to LiteRT built-in ops'),
        contains('flutter_litert 3.10.x with flutter_litert_flex 1.3.0'),
      ),
    );

    test('isAvailable is false', () {
      expect(FlexDelegate.isAvailable, isFalse);
    });

    test('create throws with the migration guidance', () async {
      await expectLater(FlexDelegate.create(), throwsA(removed));
    });

    test('the sync constructor throws with the same guidance', () {
      expect(FlexDelegate.new, throwsA(removed));
    });
  });

  group('FlexDelegate off Android', () {
    setUp(() => FlexDelegate.debugIsAndroidOverride = false);
    tearDown(() => FlexDelegate.debugIsAndroidOverride = null);

    test('a missing library is not reported as the Android removal', () {
      if (FlexDelegate.isAvailable) {
        markTestSkipped('flutter_litert_flex is bundled with this host');
        return;
      }
      expect(
        FlexDelegate.new,
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            isNot(contains('removed in flutter_litert 4.0.0')),
          ),
        ),
      );
    });
  });
}
