# Building the FlexDelegate dylib for macOS

This guide builds `libtensorflowlite_flex-mac.dylib` (arm64) from TensorFlow v2.21.0 source, plus the SwiftPM package of it, `flutter_litert_flex-macos-spm.xcframework.zip`. The dylib provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like `Conv2DBackpropFilter` and checkpoint ops like `SaveV2`/`RestoreV2`.

It is the macOS version of the Linux recipe (`build_flex_delegate_linux.md`): one monolithic shared library that exports only the two [external delegate plugin](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/lite/delegates/external/external_delegate_interface.h) entry points:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

The TF runtime, TFLite and all TF kernels are linked in statically and kept internal, so the dylib does not clash with the TFLite C library that `flutter_litert` loads next to it. Dart `dlopen`s it by path.

The build is automated in `.github/workflows/build-flex-macos.yml`. The steps below are what that workflow runs. The build inputs live in `scripts/flex/`.

## Run the workflow

```bash
gh workflow run build-flex-macos.yml --repo hugocornellier/flutter_litert --ref <branch>
```

One job on `macos-15` (about 1.5 to 3 h). It uploads:
- `libtensorflowlite_flex-mac` (dylib + `.sha256`)
- `flutter_litert_flex-macos-spm-xcframework` (zip + `.sha256`)
- `flex-macos-verify-report` (text reports)

Nothing is uploaded to a release. Publishing is a separate, manual step.

## Prerequisites (by hand)

1. **macOS on Apple Silicon** with **Xcode 16+**
2. **Bazelisk**: `brew install bazelisk`. TF 2.21.0 pins Bazel 7.7.0. `configure.py` calls `bazel --version`, so `bazel` must be on `PATH`.
3. **Python 3.9 to 3.12** with `numpy`
4. About 60 GB free disk. On GitHub runners `scripts/flex/free_macos_runner_disk.sh` frees space first.

## Step 1: Clone TensorFlow

```bash
git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git ~/tf-src
cd ~/tf-src
git rev-parse HEAD   # a481b10260dfdf833a1b16007eead49c1d7febf3
```

## Step 2: Configure

```bash
PYTHON_BIN_PATH="$(which python3)" USE_DEFAULT_PYTHON_LIB_PATH=1 \
TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_TENSORRT=0 \
TF_SET_ANDROID_WORKSPACE=0 TF_CONFIGURE_IOS=0 \
CC_OPT_FLAGS="-Wno-sign-compare" \
  python3 configure.py < /dev/null
```

## Step 3: Add the plugin wrapper and the dylib target

```bash
cp <flutter_litert>/scripts/flex/flex_delegate_plugin.cc \
  tensorflow/lite/delegates/flex/flex_delegate_plugin.cc
cat <flutter_litert>/scripts/flex/flex_BUILD_macos.append >> tensorflow/lite/delegates/flex/BUILD
```

- `flex_delegate_plugin.cc` implements the two entry points with `TFL_EXTERNAL_DELEGATE_EXPORT`. Create returns `FlexDelegate::Create().release()`, and destroy hands the pointer back to `TfLiteDelegateFactory::DeleteSimpleDelegate`, which is the deleter that unique_ptr would have run.
- `flex_BUILD_macos.append` adds `cc_binary(name = "libtensorflowlite_flex-mac.dylib", linkshared = True)` with `-Wl,-exported_symbol` for the two entry points, so nothing else is exported.

## Step 4: Build

TF v2.21.0 pins an llvm-project archive that github.com no longer serves byte-for-byte (sha256 00b1077e... instead of the pinned 3f986184...), and the TF mirror returns 404. `seed_bazel_repo_cache.sh` fetches the original from mirror.bazel.build into a repository cache, keeping it only if the sha256 is the pinned one:

```bash
bash <flutter_litert>/scripts/flex/seed_bazel_repo_cache.sh ~/bazel-repo-cache
```

```bash
bazelisk build -c opt \
  --repository_cache="$HOME/bazel-repo-cache" \
  --config=monolithic \
  --macos_minimum_os=12.0 \
  --define=tflite_convert_with_select_tf_ops=true \
  --define=with_select_tf_ops=true \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --local_ram_resources=HOST_RAM*0.7 --jobs=4 \
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex-mac.dylib
```

`--macos_minimum_os=12.0` matches TF's own `macos_arm64` release config. The 2.20.0 dylib was built without it and came out with minimum macOS 15.2, the SDK of the machine that built it.

Unlike the iOS build, this needs no apple_support patch (`scripts/patch_tf_apple_support.sh`): the flex dylib compiles no Objective-C, and run 42641596346 built without it.

TF's `.bazelrc` links macOS binaries with `-undefined dynamic_lookup`, so a missing symbol does not fail the link. The verify step checks the undefined symbols instead.

## Step 5: Strip, set the install name, sign

```bash
mkdir -p ~/flex-mac && cd ~/flex-mac
cp ~/tf-src/bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-mac.dylib .
chmod u+w libtensorflowlite_flex-mac.dylib
strip -x libtensorflowlite_flex-mac.dylib
install_name_tool -id @rpath/libtensorflowlite_flex.dylib libtensorflowlite_flex-mac.dylib
codesign -f -s - libtensorflowlite_flex-mac.dylib
shasum -a 256 libtensorflowlite_flex-mac.dylib > libtensorflowlite_flex-mac.dylib.sha256
```

The install name is the same as the 2.20.0 dylib's.

## Step 6: Make the SwiftPM xcframework

```bash
bash <flutter_litert>/scripts/flex/make_flex_macos_spm.sh libtensorflowlite_flex-mac.dylib spm
```

`flex-v1.1.0/flutter_litert_flex-macos-spm.xcframework.zip` was made by hand for `flutter_litert_flex` commit 3fdbe48, and no script existed. This script reproduces its layout: a versioned `flutter_litert_flex.framework` whose binary is an empty arm64 stub dylib (minimum macOS 11.0), with the real dylib in `Versions/A/Resources/` and the same `Info.plist` keys, wrapped with `xcodebuild -create-xcframework` and zipped with symlinks kept (`zip -ry`). SwiftPM embeds the framework, and Dart opens `Frameworks/flutter_litert_flex.framework/Versions/A/Resources/libtensorflowlite_flex-mac.dylib` by path. The workflow also checks that the zip entry list matches the `flex-v1.1.0` zip.

## Step 7: Verify

```bash
REF_DYLIB_URL=https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.0.0/libtensorflowlite_flex-mac.dylib \
REF_DYLIB_SHA256=8e08e2a9c7e1042b5ba0d309c0180e8d8d2b8a92bb65ec5164344dafd170d2e2 \
EXPECT_VERSION=2.21.0 EXPECT_MINOS=12.0 \
  bash <flutter_litert>/scripts/flex/verify_flex_macos.sh libtensorflowlite_flex-mac.dylib report
```

It reports the new and the 2.20.0 dylib side by side (size, archs, install name, minimum OS and SDK, linked libraries, exports, undefined symbols, embedded version strings, signature) and fails when:
- the arch is not arm64, or the minimum OS is not `EXPECT_MINOS`
- the exports are anything but the two plugin entry points
- an undefined symbol looks like TF, TFLite, XNNPack, Abseil or protobuf (it must all be linked in)
- it links a library outside the system
- the version string is missing, or the signature does not verify

## Step 8: Smoke test

```bash
xcrun clang -O1 -o flex_smoke_test <flutter_litert>/scripts/flex/flex_smoke_test.c
./flex_smoke_test \
  <flutter_litert>/macos/flutter_litert/Sources/flutter_litert/Resources/libtensorflowlite_c-mac.dylib \
  libtensorflowlite_flex-mac.dylib \
  <flutter_litert>/test/assets/training_model_flex.tflite
```

It loads both libraries with `dlopen`, creates the delegate through `tflite_plugin_create_delegate`, runs one step of the model's `train` signature (zero-filled inputs), destroys the delegate and prints `RESULT: train step ran`. With `--no-flex` it runs the same model without the delegate as a control. The model's only Flex ops (`FlexSaveV2`, `FlexRestoreV2`) are in its `save` and `restore` signatures, so the `train` step runs either way (the TF 2.21.0 run logged `0 nodes delegated out of 7 nodes`, and the control passed). The test proves that the dylib loads next to the TFLite C library and that the delegate is created and destroyed; it does not run a Flex kernel. In the workflow this step is informative only (`continue-on-error`), since it pairs the new dylib with whatever TFLite C library is committed.

## What this enables

Once hosted, `flutter_litert_flex`'s macOS podspec downloads `libtensorflowlite_flex-mac.dylib` and its `Package.swift` references `flutter_litert_flex-macos-spm.xcframework.zip` by URL and checksum.
