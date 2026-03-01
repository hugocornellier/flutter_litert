# Building the FlexDelegate DLL for Windows

This guide builds `libtensorflowlite_flex-win.dll` from TensorFlow v2.20.0 source. The DLL provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like `Conv2DBackpropFilter`, `Save`, `Restore`, etc.

The resulting DLL exports two symbols:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

## Prerequisites

Install these before starting:

1. **Visual Studio 2022 Build Tools** (or full VS 2022) with "Desktop development with C++" workload
2. **Bazel 6.5.0** (TF 2.20.0 requires Bazel 6.x, NOT 7.x)
   - Download from https://github.com/bazelbuild/bazel/releases/tag/6.5.0
   - Get `bazel-6.5.0-windows-x86_64.exe`, rename to `bazel.exe`, put on PATH
3. **Python 3.9–3.12** (with `numpy` installed: `pip install numpy`)
4. **MSYS2** — install to `C:\msys64`, needed for Bazel's shell tools on Windows
5. **Git for Windows**

Verify:
```powershell
bazel --version   # should show 6.5.0
python --version  # 3.9-3.12
cl                # from VS Developer Command Prompt
```

## Step 1: Clone TensorFlow

```powershell
cd C:\
git clone --depth 1 --branch v2.20.0 https://github.com/tensorflow/tensorflow.git tf-2.20.0
cd tf-2.20.0
```

## Step 2: Configure the build

Run the configure script. Answer the prompts — defaults are fine for most, just make sure Python path is correct:

```powershell
python configure.py
```

Key answers:
- Python location: wherever your Python 3.x is (e.g., `C:\Python312\python.exe`)
- CUDA support: **No** (we don't need GPU for the flex delegate)
- Everything else: defaults (just press Enter)

## Step 3: Create the C wrapper

TensorFlow's flex delegate is C++ internally. We need a thin C wrapper that exports the `tflite_plugin_create_delegate` and `tflite_plugin_destroy_delegate` symbols that flutter_litert expects.

Create the file `tensorflow/lite/delegates/flex/flex_delegate_plugin.cc`:

```cpp
// C wrapper for the TFLite Flex delegate plugin interface.
// Exports tflite_plugin_create_delegate / tflite_plugin_destroy_delegate
// as a shared library (DLL on Windows).

#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/acceleration/configuration/delegate_plugin.h"

extern "C" {

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*)) {
  // Create a default FlexDelegate. The options_keys/values are unused
  // for the flex delegate but the signature matches the plugin interface.
  auto delegate = tflite::FlexDelegate::Create();
  // Release ownership — caller is responsible for destroying via
  // tflite_plugin_destroy_delegate.
  return delegate.release();
}

EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  delete reinterpret_cast<tflite::FlexDelegate*>(delegate);
}

}  // extern "C"
```

## Step 4: Create the BUILD target

Add this to `tensorflow/lite/delegates/flex/BUILD` (append at the end of the file):

```python
# Shared library (DLL) for the Flex delegate plugin interface.
cc_binary(
    name = "libtensorflowlite_flex-win.dll",
    srcs = ["flex_delegate_plugin.cc"],
    linkshared = True,
    deps = [
        ":delegate",
        "//tensorflow/lite/acceleration/configuration:delegate_plugin",
        "//tensorflow/lite/delegates/flex:delegate_data",
        "//tensorflow/lite/delegates/flex:util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
    ],
)
```

## Step 5: Build

Open a **VS Developer Command Prompt** (or "x64 Native Tools Command Prompt for VS 2022") and run:

```powershell
cd C:\tf-2.20.0

set BAZEL_SH=C:\msys64\usr\bin\bash.exe
set BAZEL_VC=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC

bazel build -c opt --config=monolithic ^
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex-win.dll
```

Notes:
- `--config=monolithic` links everything into a single DLL (no separate TF runtime DLLs needed)
- `-c opt` enables optimizations
- Build takes 30–90 minutes depending on hardware
- If you get memory errors, add `--local_ram_resources=HOST_RAM*0.5` to limit parallelism

The output DLL will be at:
```
bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-win.dll
```

## Step 6: Verify the DLL exports

```powershell
dumpbin /exports bazel-bin\tensorflow\lite\delegates\flex\libtensorflowlite_flex-win.dll | findstr tflite_plugin
```

You should see:
```
  tflite_plugin_create_delegate
  tflite_plugin_destroy_delegate
```

Both symbols must be present.

## Step 7: Upload to GitHub Releases

Upload the DLL to the existing `flex-v1.0.0` release:

```powershell
gh release upload flex-v1.0.0 ^
  bazel-bin\tensorflow\lite\delegates\flex\libtensorflowlite_flex-win.dll ^
  --repo hugocornellier/flutter_litert ^
  --clobber
```

The `--clobber` flag overwrites if a file with the same name already exists.

## Step 8: Test

In the flutter_litert repo on Windows:

```powershell
flutter test test\native\flex_delegate_test.dart
```

All 8 tests should pass, including download, delegate creation, inference, and training.

## Troubleshooting

**Bazel version mismatch**: TF 2.20.0 requires Bazel 6.x. If you have Bazel 7.x installed, use Bazelisk (`choco install bazelisk`) and set `USE_BAZEL_VERSION=6.5.0`.

**Missing MSYS2**: Bazel needs MSYS2 for shell utilities. Set `BAZEL_SH=C:\msys64\usr\bin\bash.exe`.

**MSVC not found**: Set `BAZEL_VC` to your VS installation's VC directory. Run from a VS Developer Command Prompt.

**Out of memory during build**: Add `--local_ram_resources=HOST_RAM*0.5 --jobs=4` to the bazel command.

**Symbol not found in DLL**: Make sure the `flex_delegate_plugin.cc` wrapper uses `__declspec(dllexport)` and is compiled with `linkshared = True`.

**`configure.py` fails**: Make sure `numpy` is installed (`pip install numpy`) and Python is on PATH.

## What this enables

Once uploaded, the `flutter_litert_flex` package will download it at build time from:
```
https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.0.0/libtensorflowlite_flex-win.dll
```

It is automatically bundled into Windows app builds via CMake when `flutter_litert_flex` is added to pubspec.yaml.
