# flutter_litert_play_services_test_host

Dedicated integration test host for `flutterLitert.runtime=playServices`, the
opt-in Android build that takes LiteRT from Google Play services instead of
packaging it in the APK.

The property is set in `android/gradle.properties`. The main `example/` app
cannot be built this way because it depends on `flutter_litert_flex`, which
needs the bundled runtime, so this host depends on `flutter_litert` alone.

`integration_test/play_services_runtime_test.dart` checks that models fail
with a clear error before `initializeAndroidRuntime()`, then runs the
Interpreter and CompiledModel APIs on the Play services runtime, in helper
isolates, and side by side. The GPU test runs only when Play services provides
its GPU accelerator. It never does on emulators and not on every physical
device either, so the test skips on both CI devices and reports why.

Run on a device or emulator image with Google Play services (the runtime
module is downloaded on first use):

```bash
flutter test integration_test/play_services_runtime_test.dart -d <device>
```

Check which native libraries a build packages:

```bash
flutter build apk --debug
python3 -I ../../scripts/check_android_runtime_apk.py --runtime playServices \
  build/app/outputs/flutter-apk/app-debug.apk
```

CI runs both in `.github/workflows/flutter-ci.yml` (emulator) and
`.github/workflows/android-play-services-testlab.yml` (physical phone in
Firebase Test Lab).
