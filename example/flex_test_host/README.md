# flutter_litert_flex_test_host

Dedicated integration test host for the optional `flutter_litert_flex` addon.

This package depends on `flutter_litert` and `flutter_litert_flex` alone, so
CI can prove the addon installs, bundles, and runs on its own. The main
`example/` app also depends on `flutter_litert_flex`, because its model-matrix
tests include a Flex delegate configuration; this host is what isolates the
addon from the example's other dependencies.

Run from this directory:

```bash
flutter test -d macos integration_test
# or on Linux:
flutter test -d linux integration_test
```
