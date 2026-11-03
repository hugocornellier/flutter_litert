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
isolates, and side by side. It also runs two real networks,
face_detection_short_range and mobilefacenet (`integration_test/real_models.dart`),
on both APIs and checks their outputs against reference outputs from the
standard LiteRT CPU runtime, printing timings. The GPU tests run only when Play
services provides its GPU accelerator. It never does on emulators and not on
every physical device either, so they skip on both CI devices and report why.

`integration_test/no_play_services_test.dart` runs on a device or emulator
image without Google Play services and checks that initialization fails
promptly with both reasons.

To compare runtimes on one device, run the real-model checks on their own,
once as built and once with the bundled runtime:

```bash
flutter test integration_test/real_models_test.dart -d <device>
env 'ORG_GRADLE_PROJECT_flutterLitert.runtime=bundled' \
  flutter test integration_test/real_models_test.dart -d <device>
```

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

CI runs these in `.github/workflows/flutter-ci.yml` (emulators with and
without Play services) and
`.github/workflows/android-play-services-testlab.yml` (physical phone in
Firebase Test Lab).
