# Building the FlexDelegate Shared Library for Linux

This guide builds `libtensorflowlite_flex-linux.so` from TensorFlow v2.21.0 source. The library
provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like
`Conv2DBackpropFilter` and checkpoint ops like `SaveV2`/`RestoreV2`.

The resulting shared library exports two symbols:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

These match the [TFLite external delegate plugin interface](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/lite/delegates/external/external_delegate_interface.h).

## The CI build (preferred)

The workflow `.github/workflows/build-flex-linux.yml` does everything below on GitHub
Actions. Run it from the Actions tab (workflow_dispatch). It uses these files:

| File | What it does |
|------|--------------|
| `scripts/flex_desktop/flex_delegate_plugin.cc` | The C wrapper (Step 3) |
| `scripts/flex_desktop/flex_plugin_target.BUILD.in` | The BUILD target (Step 4), `@LIB_NAME@` is the library name |
| `scripts/flex_desktop/layout.sh` | Frees runner disk and chooses where the build goes |
| `scripts/flex_desktop/prepare_tf.sh` | Clones TF, adds the wrapper and target, runs `configure.py` (Steps 1, 2, 3, 4) |
| `scripts/flex_desktop/seed_repo_cache.sh` | Puts source archives upstream no longer serves byte-for-byte into the repository cache (see Troubleshooting) |
| `scripts/flex_desktop/build_stage.sh` | One time-boxed `bazel build` (Step 5) |
| `scripts/flex_desktop/disk_cache.sh` | Packs and unpacks the bazel caches between stages |
| `scripts/flex_desktop/verify_linux.sh`, `smoke_test.c` | Step 6, against the flex-v1.0.0 file |
| `.github/workflows/build-flex-desktop-stage.yml` | The reusable stage job shared with the Windows build |

How the workflow runs:

- **Runner**: `ubuntu-24.04`. Do not move it to a newer image without checking the GLIBC floor
  of the result (Step 6b): the library must load on every distro the 2.20.0 file loaded on.
- **Disk**: a standard runner has less free space on `/` than the build needs. `layout.sh` removes preinstalled toolchains the build never uses
  (.NET, Android SDK, GHC, CodeQL, Go, PyPy, Ruby, Node, Swift, Boost, PowerShell, Chromium,
  Julia, Miniconda, Azure CLI, Firefox and others, plus docker images and the apt cache), then
  puts the TF checkout and bazel's `--output_user_root` on whichever of `/` and `/mnt` has more
  free space, and the bazel caches on the other one when it has 40 GiB free. It also adds 8 GiB
  of swap. The job log prints `df` before and after, and every 15 minutes during the build.
- **Time**: a GitHub-hosted job may run at most 360 minutes. The v2.21.0 build took about
  3 h 10 min and finished in the first stage, but a slower runner or a later TF may not, so the
  build is split into up to five chained stage jobs. Each stage runs
  `bazel build` with `--disk_cache` and `--repository_cache`, stops bazel 35 minutes before the
  job limit, and uploads both caches as the artifact `flex-linux-hermetic-cache-stageN`
  (kept 3 days). The next stage restores them and continues; only the stage that finishes the
  build uploads the library.
- **Output**: the artifact `flex-linux-hermetic` (kept 90 days) holds
  `libtensorflowlite_flex-linux.so` (stripped), `libtensorflowlite_flex-linux.so.sha256` and
  `build-info.txt` (TF tag and commit, bazel version, runner image, exact command, size and
  sha256). The `verify` job then uploads `flex-linux-verify` (report, export lists, added and
  removed exports) and writes the report to the run summary. Nothing is uploaded to a release;
  publishing is a separate, manual step (Step 8).

The steps below are the same build done by hand.

## Prerequisites

1. **Bazelisk**, which reads `.bazelversion` from the TF source (TF v2.21.0 pins
   **Bazel 7.7.0**):
   ```bash
   mkdir -p ~/.local/bin
   curl -Lo ~/.local/bin/bazel \
     https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
   chmod +x ~/.local/bin/bazel
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Python 3** with **numpy**, required by `configure.py` (`pip3 install numpy`). The build
   itself uses TF's hermetic Python 3.11.

3. **Clang** installed locally (`sudo apt-get install clang`). TF v2.21.0 builds Linux targets
   with its own hermetic clang toolchain (`USE_HERMETIC_CC_TOOLCHAIN=1` in TF's `.bazelrc`), but
   `configure.py` still asks for a local clang path.

4. About **50 GiB of free disk space** (bazel output tree, caches, source and output). Put it on
   one large volume with `--output_user_root` if your home directory is small.

## Step 1: Clone TensorFlow

```bash
git clone --depth 1 --branch v2.21.0 \
  https://github.com/tensorflow/tensorflow.git tf-2.21.0
cd tf-2.21.0
cat .bazelversion   # 7.7.0
```

## Step 2: Configure the Build

The workflow answers every prompt through environment variables:

```bash
export TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_TENSORRT=0
export TF_SET_ANDROID_WORKSPACE=0 TF_CONFIGURE_IOS=0
export CC_OPT_FLAGS="-Wno-sign-compare"
export USE_DEFAULT_PYTHON_LIB_PATH=1
export PYTHON_BIN_PATH="$(command -v python3)"
export TF_NEED_CLANG=1 CLANG_COMPILER_PATH="$(command -v clang)"
yes "" | python3 configure.py
```

This writes `.tf_configure.bazelrc`. `CC_OPT_FLAGS` only applies under `--config=opt`, which
the build does not use, so no `-march=native` ends up in the library.

## Step 3: Create the C Wrapper

Copy `scripts/flex_desktop/flex_delegate_plugin.cc` from this repo to
`tensorflow/lite/delegates/flex/flex_delegate_plugin.cc`. It is the same file for Linux and
Windows:

```cpp
#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

extern "C" {

TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys, const char* const* options_values,
    size_t num_options, void (*report_error)(const char*)) {
  return tflite::FlexDelegate::Create().release();
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

}  // extern "C"
```

`TFL_EXTERNAL_DELEGATE_EXPORT` (from `external_delegate_interface.h`) is
`__attribute__((visibility("default")))` on Linux.

> **Why not `delete reinterpret_cast<tflite::FlexDelegate*>(delegate)`?**
> `FlexDelegate::Create()` returns `TfLiteDelegateUniquePtr`, a
> `std::unique_ptr<TfLiteDelegate, void(*)(TfLiteDelegate*)>` with the custom
> deleter `TfLiteDelegateFactory::DeleteSimpleDelegate`. Plain `delete` would skip
> that deleter and corrupt memory.

## Step 4: Add the Bazel BUILD Target

Append `scripts/flex_desktop/flex_plugin_target.BUILD.in` to
`tensorflow/lite/delegates/flex/BUILD`, with `@LIB_NAME@` replaced:

```bash
sed 's/@LIB_NAME@/libtensorflowlite_flex-linux.so/g' \
  /path/to/flutter_litert/scripts/flex_desktop/flex_plugin_target.BUILD.in \
  >> tensorflow/lite/delegates/flex/BUILD
```

which appends:

```python
cc_binary(
    name = "libtensorflowlite_flex-linux.so",
    srcs = ["flex_delegate_plugin.cc"],
    linkshared = True,
    local_defines = ["TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY"],
    deps = [
        ":delegate",
        "//tensorflow/lite/delegates/external:external_delegate_interface",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
    ],
)
```

`TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY` makes the export macro mean "export" rather than
"import" on Windows; it has no effect on Linux but keeps the target identical on both.

## Step 5: Build

```bash
bazel --output_user_root=/big/disk/bazel build \
  -c opt \
  --config=monolithic \
  --define=tflite_convert_with_select_tf_ops=true \
  --define=with_select_tf_ops=true \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --local_ram_resources=HOST_RAM*.75 \
  --disk_cache=/big/disk/cache/disk \
  --repository_cache=/big/disk/cache/repo \
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex-linux.so
```

**Flags:**
- `-c opt`: optimized build. `--config=opt` is deliberately not used (see Step 2).
- `--config=monolithic`: statically links all TF/TFLite deps, so the `.so` is self-contained.
- `--define=tflite_convert_with_select_tf_ops=true`, `--define=with_select_tf_ops=true`:
  register and link the full set of TF ops.
- `--repo_env=HERMETIC_PYTHON_VERSION=3.11`: the hermetic Python the build uses.
- `--local_ram_resources=HOST_RAM*.75`: avoids the OOM killer on a 16 GiB machine.
- `--disk_cache`, `--repository_cache`: optional by hand; the CI stages need them to resume.

The output is at `bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so`.
Copy it out and strip it, as the 2.20.0 file was (the dynamic symbols stay):

```bash
cp bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so .
chmod u+w libtensorflowlite_flex-linux.so
strip --strip-unneeded libtensorflowlite_flex-linux.so
```

The v2.21.0 file is about 466 MB unstripped and about 368 MB stripped (367819776 bytes); the
2.20.0 file was about 349 MB stripped. The CI build does the strip in `build_stage.sh` and records it in `build-info.txt`.

## Step 6: Verify the Library

`scripts/flex_desktop/verify_linux.sh <new.so> <old.so> <out_dir>` does all of this and
compares against the file it replaces (`flex-v1.0.0`, downloaded and only inspected):

### 6a. The two plugin symbols are exported
```bash
nm -D --defined-only libtensorflowlite_flex-linux.so | grep tflite_plugin
```
Both must show `T` (or `W`).

### 6b. Architecture and GLIBC floor
```bash
file libtensorflowlite_flex-linux.so          # ELF 64-bit LSB shared object, x86-64
objdump -T libtensorflowlite_flex-linux.so | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1
objdump -p libtensorflowlite_flex-linux.so | grep NEEDED
```
The highest `GLIBC_` version must not be higher than the old file's. `NEEDED` lists only
system libraries (`libstdc++`, `libm`, `libc`, `libgcc_s`, and so on), never a TF library.

### 6c. Version
```bash
grep -a -oE '2\.21\.[0-9]+' libtensorflowlite_flex-linux.so | sort | uniq -c
```

### 6d. Exports compared with the old file
`verify_linux.sh` writes `exports-added.txt` and `exports-removed.txt` (from
`nm -D --defined-only`) and prints the counts.

### 6e. Smoke test
`smoke_test.c` loads the library with `dlopen`, creates and destroys one delegate, and prints
`SMOKE OK`.

## Step 7: Test Locally

```bash
mkdir -p ~/.cache/flutter_litert
cp libtensorflowlite_flex-linux.so ~/.cache/flutter_litert/

cd /path/to/flutter_litert
flutter test test/native/flex_delegate_test.dart
flutter test test/native/flex_save_restore_test.dart
```

All `flex_delegate_test.dart` tests should pass. The `flex_save_restore_test.dart` subprocess
exits with SIGBUS (TF atexit handler crash); this is expected and handled by the test wrapper.

## Step 8: Publish (manual, not done by the workflow)

The workflow never creates or changes a release. To ship a new build, upload the artifact's
`.so` to a new `flex-vX.Y.Z` release in this repo, then point `flutter_litert_flex`'s Linux
CMake download URL and its SHA-256 at it.

## Troubleshooting

**Bazel version mismatch**: use Bazelisk; it reads `.bazelversion` (7.7.0 for TF v2.21.0).

**`Error downloading ... llvm-project ... Checksum was 00b1077e... but wanted 3f986184...`**:
github.com no longer serves the llvm-project archive TF v2.21.0 pins byte-for-byte, and the TF
mirror returns 404. Run `scripts/flex_desktop/seed_repo_cache.sh <dir>` and build with
`--repository_cache=<dir>`: it fetches the archive from mirror.bazel.build and keeps it only if
its sha256 is the pinned one.

**numpy not found during configure**: `pip3 install numpy`.

**Out of memory**: lower `--local_ram_resources` or add `--jobs=4`, or add swap.

**Disk full**: point `--output_user_root` and the caches at a larger volume.
`bazel clean --expunge` frees the output tree.

**`tflite_plugin_*` symbols not found**: check the wrapper is in place and the target was
appended: `grep -n "libtensorflowlite_flex-linux" tensorflow/lite/delegates/flex/BUILD`.

**`NEEDED` shows TF libraries**: `--config=monolithic` was not applied.

## Key Differences from the Windows Build

| Aspect | Linux | Windows |
|--------|-------|---------|
| Workflow | `build-flex-linux.yml` | `build-flex-windows.yml` |
| Runner | `ubuntu-24.04` | `windows-2022` |
| Output name | `libtensorflowlite_flex-linux.so` | `libtensorflowlite_flex-win.dll` |
| Export macro | `TFL_EXTERNAL_DELEGATE_EXPORT` (visibility default) | `TFL_EXTERNAL_DELEGATE_EXPORT` (`__declspec(dllexport)` with `TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY`) |
| Compiler | TF's hermetic clang | MSVC (VS 2022), clang-cl as a fallback |
| Shell tools | Native bash | MSYS2 (`BAZEL_SH`) |
| Symbol check | `nm -D` | `dumpbin /exports` |
| Cache path | `~/.cache/flutter_litert/` | `%LOCALAPPDATA%\flutter_litert\cache\` |
