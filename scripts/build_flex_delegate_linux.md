# Building the FlexDelegate Shared Library for Linux

This guide builds `libtensorflowlite_flex-linux.so` from TensorFlow v2.20.0 source. The library
provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like
`Conv2DBackpropFilter`, `Save`, `Restore`, etc.

The resulting shared library exports two symbols:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

These match the [TFLite external delegate plugin interface](https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/lite/delegates/external/external_delegate_interface.h).

## Prerequisites

Install these before starting:

1. **Bazelisk** — manages Bazel versions automatically; reads `.bazelversion` from TF source
   (TF v2.20.0 requires **Bazel 7.4.1**; Bazelisk fetches it automatically):
   ```bash
   mkdir -p ~/.local/bin
   curl -Lo ~/.local/bin/bazel \
     https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
   chmod +x ~/.local/bin/bazel
   export PATH="$HOME/.local/bin:$PATH"
   ```
   Add the export to `~/.bashrc` or `~/.profile` for persistence.

2. **Python 3** with **numpy** — required by `configure.py`:
   ```bash
   # Ubuntu/Debian (with pip available):
   pip3 install numpy
   # Or via apt:
   sudo apt-get install python3-numpy
   ```

3. **Clang** (recommended) or **GCC 9+** — the build system auto-detects whichever is
   available. Clang 17/18 is the officially preferred compiler:
   ```bash
   sudo apt-get install clang   # usually sufficient; any recent version works
   ```
   GCC 13 also works fine if Clang is not present.

4. Ensure **~50 GiB of free disk space** (Bazel cache + source + output).

Verify:
```bash
bazel --version   # Should output: "Bazel 7.4.1" (fetched automatically)
python3 --version # 3.9–3.13
python3 -c "import numpy; print(numpy.__version__)"
gcc --version     # or clang --version
```

## Step 1: Clone TensorFlow

```bash
cd /tmp
git clone --depth 1 --branch v2.20.0 \
  https://github.com/tensorflow/tensorflow.git tf-2.20.0
cd tf-2.20.0
```

Estimated download: ~2–3 GiB with `--depth 1`.

## Step 2: Configure the Build

```bash
python3 configure.py
```

**Answers to prompts:**
- Python location: press **Enter** (accept default)
- Python library path: press **Enter** (accept default)
- ROCm support: **n**
- CUDA support: **n**
- Clang path: press **Enter** (accept default auto-detected path)
- Optimization flags: press **Enter** (accept default)
- Android builds: **n**

This generates `.tf_configure.bazelrc` with your system's paths.

## Step 3: Create the C Wrapper

Create `tensorflow/lite/delegates/flex/flex_delegate_plugin.cc`:

```cpp
// C wrapper for the TFLite Flex delegate external plugin interface.
//
// Exports the two symbols required by flutter_litert's FlexDelegate loader:
//   tflite_plugin_create_delegate
//   tflite_plugin_destroy_delegate
//
// Built into libtensorflowlite_flex-linux.so with --config=monolithic so it is
// fully self-contained (no external TF/TFLite runtime dependencies at load
// time). Symbol visibility follows the standard external delegate interface
// header (external_delegate_interface.h):
//   - Linux: __attribute__((visibility("default"))) via TFL_EXTERNAL_DELEGATE_EXPORT
//
// FlexDelegate::Create() returns TfLiteDelegateUniquePtr, which carries a
// custom deleter (TfLiteDelegateFactory::DeleteSimpleDelegate). We must use
// that deleter — never plain `delete` — when destroying the delegate.

#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

extern "C" {

TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*)) {
  // FlexDelegate::Create() returns a TfLiteDelegateUniquePtr (unique_ptr with
  // custom deleter). We release the raw pointer here; ownership is transferred
  // to the caller and must be freed via tflite_plugin_destroy_delegate.
  auto delegate = tflite::FlexDelegate::Create();
  return delegate.release();
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  // Use the correct deleter for a SimpleDelegateInterface-based delegate.
  // This matches the deleter stored in the TfLiteDelegateUniquePtr returned
  // by FlexDelegate::Create().
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

}  // extern "C"
```

> **Why not `delete reinterpret_cast<tflite::FlexDelegate*>(delegate)`?**
> `FlexDelegate::Create()` returns `TfLiteDelegateUniquePtr` — a
> `std::unique_ptr<TfLiteDelegate, void(*)(TfLiteDelegate*)>` with the custom
> deleter `TfLiteDelegateFactory::DeleteSimpleDelegate`. Using plain `delete`
> would skip this custom deleter and corrupt memory.

## Step 4: Add the Bazel BUILD Target

Append to `tensorflow/lite/delegates/flex/BUILD`:

```python
# ---------------------------------------------------------------------------
# flutter_litert: FlexDelegate external plugin shared library for Linux.
#
# Exports the two symbols required by the external delegate plugin interface:
#   tflite_plugin_create_delegate
#   tflite_plugin_destroy_delegate
#
# Build with:
#   bazel build -c opt --config=monolithic \
#     //tensorflow/lite/delegates/flex:libtensorflowlite_flex-linux.so
# ---------------------------------------------------------------------------
cc_binary(
    name = "libtensorflowlite_flex-linux.so",
    srcs = ["flex_delegate_plugin.cc"],
    linkshared = True,
    deps = [
        ":delegate",
        "//tensorflow/lite/delegates/external:external_delegate_interface",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
    ],
)
```

## Step 5: Build

```bash
cd /tmp/tf-2.20.0
export PATH="$HOME/.local/bin:$PATH"

bazel build \
  -c opt \
  --config=monolithic \
  --define=tflite_convert_with_select_tf_ops=true \
  --define=with_select_tf_ops=true \
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex-linux.so
```

**Flag explanations:**
- `-c opt` — optimized build (essential for production size and performance)
- `--config=monolithic` — statically links all TF/TFLite deps; makes the `.so` fully
  self-contained with no external runtime dependencies
- `--define=tflite_convert_with_select_tf_ops=true` — enables SELECT_TF_OPS registration
- `--define=with_select_tf_ops=true` — links the full set of TF ops into the library

**If RAM is tight** (< 8 GiB free), add:
```bash
--local_ram_resources=6144 --jobs=4
```

**Estimated build time:** 45–90 minutes on a modern machine (12 cores).
**Estimated Bazel cache size:** 20–40 GiB.

The output will be at:
```
bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so
```

## Step 6: Verify the Library

### 6a. Confirm the two required symbols are exported:
```bash
nm -D bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so \
  | grep tflite_plugin
```

Expected output:
```
<address> T tflite_plugin_create_delegate
<address> T tflite_plugin_destroy_delegate
```

Both must appear with `T` (defined in the text/code section, publicly exported).
Any other status letter or a missing symbol means the build needs investigation.

### 6b. Confirm the library is self-contained:
```bash
ldd bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so
```

Should only list standard system libraries: `libstdc++.so`, `libm.so`, `libc.so`,
`libdl.so`, `libpthread.so`. Must **not** list `libtensorflowlite.so` or any other
TF-specific library — those are statically linked inside.

### 6c. Check the file size:
```bash
ls -lh bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so
```

Expected: ~100–160 MB (comparable to macOS `libtensorflowlite_flex-mac.dylib` at ~123 MB).

## Step 7: Test Locally

Place the library in the flutter_litert cache directory and run the tests:

```bash
mkdir -p ~/.cache/flutter_litert
cp bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so \
   ~/.cache/flutter_litert/

cd /path/to/flutter_litert
flutter test test/native/flex_delegate_test.dart
flutter test test/native/flex_save_restore_test.dart
```

All `flex_delegate_test.dart` tests should pass. The `flex_save_restore_test.dart` subprocess
exits with SIGBUS (TF atexit handler crash) — this is expected and handled by the test wrapper;
the test passes as long as no `Expected:` assertion failure lines appear in the output.

## Step 8: Upload to GitHub Releases

```bash
gh release upload flex-v1.0.0 \
  bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-linux.so \
  --repo hugocornellier/flutter_litert \
  --clobber
```

Verify the release now lists the asset:
```bash
gh release view flex-v1.0.0 --repo hugocornellier/flutter_litert
```

Once uploaded, the `flutter_litert_flex` package will download it at build time from:
```
https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.0.0/libtensorflowlite_flex-linux.so
```

It is automatically bundled into Linux app builds via CMake when `flutter_litert_flex` is added to pubspec.yaml.

## Troubleshooting

**Bazel version mismatch**: If `bazel --version` doesn't show 7.4.1, make sure you installed
Bazelisk (not a direct Bazel binary) and that `~/.local/bin` is on your PATH.

**numpy not found during configure**: Install with `pip3 install numpy` or
`sudo apt-get install python3-numpy`.

**Out of memory during build**: Add `--local_ram_resources=HOST_RAM*0.5 --jobs=4` to limit
parallelism. On a 16 GiB machine, `--local_ram_resources=8192` is safe.

**Disk full during build**: Bazel's cache can grow to 40+ GiB. Free up disk or redirect the
cache with `--output_user_root=/path/to/large/disk`. Clean with `bazel clean --expunge` and
restart (this clears the cache and adds build time).

**`tflite_plugin_*` symbols not found**: Make sure `flex_delegate_plugin.cc` is correctly
saved and the BUILD target is properly appended. Re-check with
`grep -n "libtensorflowlite_flex-linux" tensorflow/lite/delegates/flex/BUILD`.

**`ldd` shows unexpected TF libraries**: The `--config=monolithic` flag was not applied.
Rebuild with the flag to statically link all dependencies.

**Symbol shows `W` instead of `T` in nm output**: Weak symbols (`W`) are also valid for
linker resolution on Linux. The delegate will still be loaded and called correctly.

## Key Differences from the Windows Build

| Aspect | Linux | Windows |
|--------|-------|---------|
| Output name | `libtensorflowlite_flex-linux.so` | `libtensorflowlite_flex-win.dll` |
| Export macro | `__attribute__((visibility("default")))` (via `TFL_EXTERNAL_DELEGATE_EXPORT`) | `__declspec(dllexport)` |
| Shell tools | Native Bash | Requires MSYS2 |
| Compiler | Clang (auto-detected) or GCC | MSVC |
| Bazel shell | No extra env vars needed | `BAZEL_SH=C:\msys64\usr\bin\bash.exe` |
| Symbol verify | `nm -D libfile.so \| grep tflite_plugin` | `dumpbin /exports libfile.dll \| findstr tflite_plugin` |
| Dep verify | `ldd libfile.so` | Dependency Walker |
| Cache path | `~/.cache/flutter_litert/` | `%LOCALAPPDATA%\flutter_litert\cache\` |
