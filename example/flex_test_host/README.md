# flutter_litert_flex_test_host

Dedicated integration test host for the optional `flutter_litert_flex` addon.

This package depends on `flutter_litert` and `flutter_litert_flex` alone, so
CI can prove the addon installs, bundles, and runs on its own.

It has no Android app on purpose. Android flex was removed in flutter_litert
4.0.0 because LiteRT 2.x has no flex delegate, and the Android build fails
when `flutter_litert_flex` is included. Do not add an Android folder here.

Run from this directory:

```bash
flutter test -d macos integration_test
# or on Linux:
flutter test -d linux integration_test
# or on Windows:
flutter test -d windows integration_test
```
