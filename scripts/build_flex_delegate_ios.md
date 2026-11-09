# Building the FlexDelegate xcframework for iOS

This guide builds `TensorFlowLiteFlex.xcframework` from TensorFlow v2.21.0 source. The xcframework provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like `Conv2DBackpropFilter` and checkpoint ops like `SaveV2`/`RestoreV2`.

The build is automated in `.github/workflows/build-flex-ios.yml` (dispatch only). The steps below are what that workflow runs, so they can also be followed by hand. The build inputs live in `scripts/flex/`.

The resulting xcframework exports two plugin entry points:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

**Important**: iOS does not allow `dlopen` for third-party dynamic libraries. The flex delegate is built as a **static framework** and linked into the app binary at build time. At runtime, Dart finds the symbols via `DynamicLibrary.process()`.

Two forms are produced, both used by `flutter_litert_flex`:

| File | Form | Used by |
| --- | --- | --- |
| `TensorFlowLiteFlex-ios.xcframework.zip` | framework xcframework, static archive per slice | the CocoaPods podspec |
| `TensorFlowLiteFlex-spm.xcframework.zip` | library xcframework, `libTensorFlowLiteFlex.a` per slice, XNNPack objects removed | `Package.swift` |

Slices: `ios-arm64` and `ios-arm64-simulator`. There is no x86_64 simulator slice (the 2.20.0 release had none either).

## Run the workflow

```bash
gh workflow run build-flex-ios.yml --repo hugocornellier/flutter_litert --ref <branch>
```

Optional inputs `tflc_spm_url` and `tflc_spm_sha256` pick the `TensorFlowLiteC` SPM xcframework that the verify step links against (default: the `flex-v1.1.1` one). Point them at the matching TFLite C build to check the pair.

Jobs:
1. `build (ios-arm64)` and `build (ios-arm64-simulator)` in parallel on `macos-15`, about 1 h each. Each uploads its framework zip as a short-lived artifact.
2. `package` assembles the xcframework, makes the SPM form, verifies both and uploads:
   - `TensorFlowLiteFlex-ios-xcframework` (zip + `.sha256`)
   - `TensorFlowLiteFlex-spm-xcframework` (zip + `.sha256`)
   - `flex-ios-verify-report` (text reports and symbol lists)

If only verification fails, `gh run rerun <id> --failed` re-runs `package` without rebuilding.

Nothing is uploaded to a release. Publishing is a separate, manual step.

## Prerequisites (by hand)

1. **macOS** with **Xcode 16+** (iOS SDK and command-line tools)
2. **Bazelisk**: `brew install bazelisk`. TF 2.21.0 pins Bazel 7.7.0 in `.bazelversion`. Set `USE_BAZEL_VERSION=7.7.0` to be explicit. `configure.py` calls `bazel --version`, so `bazel` must be on `PATH` (Homebrew's bazelisk installs it as `bazel` too).
3. **Python 3.9 to 3.12** with `numpy`
4. About 60 GB free disk per slice build. On GitHub runners, `scripts/flex/free_macos_runner_disk.sh` frees space first (simulator runtimes, Android SDK, .NET and similar). It keeps every Xcode unless `REMOVE_OTHER_XCODES=1`.

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
TF_SET_ANDROID_WORKSPACE=0 TF_CONFIGURE_IOS=1 \
CC_OPT_FLAGS="-Wno-sign-compare" \
  python3 configure.py < /dev/null
```

`TF_CONFIGURE_IOS=1` makes `tensorflow/lite/ios/BUILD` a symlink to `BUILD.apple`.

## Step 3: Add the plugin wrapper and the framework target

```bash
cp <flutter_litert>/scripts/flex/flex_delegate_plugin_ios.cc \
  tensorflow/lite/delegates/flex/flex_delegate_plugin.cc
cat <flutter_litert>/scripts/flex/flex_BUILD_ios.append >> tensorflow/lite/delegates/flex/BUILD
cat <flutter_litert>/scripts/flex/BUILD_apple_flex.append >> tensorflow/lite/ios/BUILD.apple
```

- `flex_delegate_plugin_ios.cc` implements the two entry points. `FlexDelegate::Create()` returns a `TfLiteDelegateUniquePtr` with a custom deleter, so the wrapper keeps each unique_ptr in a map and `tflite_plugin_destroy_delegate` erases it, which runs that deleter. (Older versions of this guide used `delete reinterpret_cast<tflite::FlexDelegate*>(delegate)`, which is wrong: the returned pointer is a `TfLiteDelegate`, not a `FlexDelegate`.)
- `flex_BUILD_ios.append` adds `cc_library(name = "flex_delegate_plugin", alwayslink = True)`.
- `BUILD_apple_flex.append` adds `ios_static_framework(name = "TensorFlowLiteFlex_framework")`. Its `avoid_deps` keep the core TFLite C types out, as `TensorFlowLiteC` provides them, and it uses `minimum_os_version = TFL_MINIMUM_OS_VERSION` (12.0 in 2.21.0).

## Step 4: Build each slice

TF v2.21.0 pins an llvm-project archive that github.com no longer serves byte-for-byte (sha256 00b1077e... instead of the pinned 3f986184...), and the TF mirror returns 404. Seed a repository cache with the original from mirror.bazel.build first (the script keeps it only if the sha256 is the pinned one):

```bash
bash <flutter_litert>/scripts/flex/seed_bazel_repo_cache.sh ~/bazel-repo-cache
```

Patch apple_support so Objective-C(++) compiles get the Xcode environment. TF 2.21.0 moved to apple_support 1.24.5, whose crosstool scopes `-c <src>`, `-target` and `XCODE_VERSION_OVERRIDE` to `all_cc_compile_actions`; TF's WORKSPACE resolves a rules_cc older than 0.2 that leaves `objc-compile` and `objc++-compile` out of that group, so without the patch the first `.mm` compile (`tensorflow/lite/profiling/signpost_profiler.mm`) fails with `wrapped_clang: Error: DEVELOPER_DIR not set.` The script adds `patches/apple_support_objc_compile_actions.patch` as a `patch_file` on the apple_support archive (the same patch `build-tflite-ios.yml` uses):

```bash
sh <flutter_litert>/scripts/patch_tf_apple_support.sh ~/tf-src
```

Do not pass `--action_env=SDKROOT` or `--action_env=DEVELOPER_DIR` instead: Bazel adds both itself for actions that carry the Xcode env, and the duplicate key crashes Bazel 7.7.0 (`IllegalArgumentException: Multiple entries with same key: SDKROOT=...`).

Then build, passing `--repository_cache="$HOME/bazel-repo-cache"` to each `bazelisk build`:

```bash
# device
bazelisk build -c opt --repository_cache="$HOME/bazel-repo-cache" --config=ios --cpu=ios_arm64 \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --local_ram_resources=HOST_RAM*0.7 --jobs=4 \
  //tensorflow/lite/ios:TensorFlowLiteFlex_framework
cp bazel-bin/tensorflow/lite/ios/TensorFlowLiteFlex_framework.zip ~/flex-ios-arm64.zip

# simulator (Apple Silicon)
bazelisk build -c opt --repository_cache="$HOME/bazel-repo-cache" --config=ios --cpu=ios_sim_arm64 \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --local_ram_resources=HOST_RAM*0.7 --jobs=4 \
  //tensorflow/lite/ios:TensorFlowLiteFlex_framework
cp bazel-bin/tensorflow/lite/ios/TensorFlowLiteFlex_framework.zip ~/flex-ios-arm64-simulator.zip
```

## Step 5: Package the CocoaPods xcframework

```bash
mkdir -p ~/flex-ios/{ios-arm64,ios-arm64-simulator}
unzip -q ~/flex-ios-arm64.zip -d ~/flex-ios/ios-arm64
unzip -q ~/flex-ios-arm64-simulator.zip -d ~/flex-ios/ios-arm64-simulator
xcodebuild -create-xcframework \
  -framework ~/flex-ios/ios-arm64/TensorFlowLiteFlex.framework \
  -framework ~/flex-ios/ios-arm64-simulator/TensorFlowLiteFlex.framework \
  -output ~/flex-ios/TensorFlowLiteFlex.xcframework
cd ~/flex-ios
zip -qr TensorFlowLiteFlex-ios.xcframework.zip TensorFlowLiteFlex.xcframework
shasum -a 256 TensorFlowLiteFlex-ios.xcframework.zip > TensorFlowLiteFlex-ios.xcframework.zip.sha256
```

## Step 6: Make the SwiftPM form

```bash
FLEX_REPO_COMMIT=af65014505cc1fd0b0e959de0dfaf86c170b6239 \
PREPROCESS_SHA256=5fa7fcfcd6591109d326188f6a5ecaf7f26c67bf2c096a82cfcbdf43dc4a3a04 \
  bash <flutter_litert>/scripts/flex/make_flex_ios_spm.sh \
  ~/flex-ios/TensorFlowLiteFlex-ios.xcframework.zip ~/flex-ios/spm
```

This runs `flutter_litert_flex`'s own `scripts/preprocess_flex_xcframework.sh` (fetched at the pinned commit and checked against the sha256) on the local zip instead of on a release download, with one more line changed: its dedup pattern for Bazel's hash-suffixed object copies also accepts 16 hex digits (see below). That script extracts each slice, drops each hash-suffixed copy of a same-named object whose strong symbols another copy already defines, drops objects that need WebP, ICU or protobuf-compiler symbols, removes the objects that define XNNPack symbols (`TensorFlowLiteC` provides them under SwiftPM, where the podspec's `nmedit` step does not run) and repacks. The result is then repackaged as a `-library` xcframework with `libTensorFlowLiteFlex.a` per slice, the same layout as `flex-v1.1.1/TensorFlowLiteFlex-spm.xcframework.zip`.

Why the pattern change: when two Bazel targets compile a source file with the same base name, the framework archive holds both objects, the second one renamed with a hash suffix. Bazel 7.7.0 with TF 2.21.0 names them like `node_def_util_3e6b16f69e25a5d1.o` (16 hex). At the pinned commit the preprocess script and the podspec's Ruby copy of the same dedup match only `_[a-f0-9]{32}\.o`, so on a 2.21.0 archive they keep both copies and an `-all_load` or `-force_load` link fails with about 2600 duplicate symbols (seen in run 42656958304). `flutter_litert_flex` 2.1.0 has the same change in its `preprocess_flex_xcframework.sh` and in its iOS podspec (flex commit ea29249).

About the pin: af65014 is the commit the `flex-v2.0.0` files were built with, so it stays the default. `make_flex_ios_spm.sh` also accepts a newer flex commit: when the script already has the widened pattern it leaves it alone, and because the flex script checks its download against `DOWNLOAD_SHA256` from 563e121 on, it is run with `DOWNLOAD_SHA256` set to the local zip's own hash. A newer pin needs its own `PREPROCESS_SHA256`. Only the af65014 pin has been run in CI.

## Step 7: Verify

```bash
bash <flutter_litert>/scripts/flex/verify_flex_ios.sh \
  ~/flex-ios/TensorFlowLiteFlex-ios.xcframework.zip \
  ~/flex-ios/spm/TensorFlowLiteFlex-spm.xcframework.zip \
  ~/flex-ios/report
```

Needs the `REF_*`, `TFLC_SPM_*` and `EXPECT_VERSION` environment variables (see the workflow's `env`). It checks and reports:
- per slice: archs (`lipo -archs`), minimum OS (`LC_BUILD_VERSION`), size and sha256, next to the 2.20.0 release
- the TF version string embedded in the binary
- `tflite_plugin_create_delegate` and `tflite_plugin_destroy_delegate` are defined in every slice of both forms (hard failure if not)
- defined global symbols diffed against 2.20.0, for both forms (`scripts/flex/flex_symbols.py`)
- the 15 symbols that the `flutter_litert` podspec hides from `TensorFlowLiteC` with `nmedit -R`, against the actual strong-symbol overlap between this framework and `TensorFlowLiteC`
- the TFLite symbols this framework expects to be provided by `TensorFlowLiteC`, and whether the given `TensorFlowLiteC` defines them all
- how many object members carry a 16- or 32-digit hash suffix, next to the 2.20.0 archives
- a SwiftPM-style link test (`-all_load` with `TensorFlowLiteC`) for both slices and, as a control, for the 2.20.0 SPM device slice. Duplicate symbols in the new links are a hard failure; undefined symbols are reported only

## Troubleshooting

**Bazel version mismatch**: use `USE_BAZEL_VERSION=7.7.0 bazelisk`.

**Xcode SDK not found**: `sudo xcode-select -s /Applications/Xcode.app`.

**`wrapped_clang: DEVELOPER_DIR not set`** on an `.mm` compile: the apple_support patch in Step 4 is missing.

**`ld: ... duplicate symbols` in the SwiftPM-form link test**: the dedup pattern in Step 6 did not match Bazel's hash suffix. Compare the suffix counts in the report with the pattern.

**Build fails with "undeclared inclusion"**: run `configure.py` first.

**Out of memory or disk**: lower `--jobs` and `--local_ram_resources`, and free disk before building.

**`configure.py` fails or waits for input**: use Python 3.9 to 3.12 with `numpy`, set `USE_DEFAULT_PYTHON_LIB_PATH=1` and feed it `< /dev/null`.

**Bitcode warnings**: `--config=ios` still passes `-fembed-bitcode`. The warnings are harmless. If a newer Xcode turns them into errors, add `--copt=-fno-embed-bitcode`.

## What this enables

Once hosted, the `flutter_litert_flex` podspec downloads `TensorFlowLiteFlex-ios.xcframework.zip` and `Package.swift` references `TensorFlowLiteFlex-spm.xcframework.zip` by URL and checksum. Under CocoaPods the xcframework is vendored and force-loaded. Under SwiftPM it is linked with `-all_load`. At runtime the flex delegate symbols are in the app binary, and Dart finds them via `DynamicLibrary.process()`.
