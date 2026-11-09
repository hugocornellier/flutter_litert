# Building the FlexDelegate DLL for Windows

This guide builds `libtensorflowlite_flex-win.dll` from TensorFlow v2.21.0 source. The DLL
provides `SELECT_TF_OPS` support for on-device training models that use gradient ops like
`Conv2DBackpropFilter` and checkpoint ops like `SaveV2`/`RestoreV2`.

The resulting DLL exports two symbols:
- `tflite_plugin_create_delegate`
- `tflite_plugin_destroy_delegate`

## The CI build (preferred)

The workflow `.github/workflows/build-flex-windows.yml` does everything below on GitHub
Actions. Run it from the Actions tab (workflow_dispatch). It shares its scripts with the Linux
build (see `build_flex_delegate_linux.md` for the file list); the Windows-only ones are
`scripts/flex_desktop/verify_windows.sh` and the Windows branches of `layout.sh`,
`prepare_tf.sh` and `build_stage.sh`.

How the workflow runs:

- **Runner**: `windows-2022` (Visual Studio 2022, MSYS2 at `C:\msys64`). All steps run in Git
  Bash. The clang chain installs LLVM 19.1.7 to `C:\llvm19` (the runner's LLVM 20 is not
  used), and both chains pin the Windows SDK to the newest one older than 10.0.26100 that the
  runner has (`BAZEL_WINSDK_FULL_VERSION`, see Troubleshooting).
- **Compiler**: the `clang` chain (clang-cl, the only compiler TF upstream builds Windows with)
  makes the DLL to ship; it finishes in one stage, about 2 h 15 min. The `msvc` chain (cl.exe,
  what the 2.20.0 DLL was built with) does not build TF v2.21.0: after the four MSVC fixes
  below it still fails inside LLVM (`llvm/lib/Support/TrieRawHashMap.cpp: error C2990:
  'llvm::TrailingObjects': non-class template has already been declared as a class template`).
  It only runs when the workflow is started with the `msvc` input set, for trying a later TF.
- **Disk**: `layout.sh` turns off Defender real-time scanning (it slows bazel a lot), then puts
  the TF checkout and bazel's `--output_user_root` on whichever of `C:` and `D:` (when the
  runner has one) has more free space, under a short path (`C:\fw` or `D:\fw`) to stay clear of
  the 260-character path limit. The bazel caches go on the other drive when it has 40 GiB free.
  The job log prints the drives before the build and `df` every 15 minutes during it.
- **Time**: a GitHub-hosted job may run at most 360 minutes. Each chain is split into up to six
  stage jobs. Each stage runs `bazel build` with `--disk_cache` and `--repository_cache`, stops
  bazel 55 minutes before the job limit, and uploads both caches as the artifact
  `flex-windows-<compiler>-cache-stageN` (kept 3 days). The next stage restores them and
  continues; only the stage that finishes the build uploads the DLL.
- **Output**: the artifact `flex-windows-clang` (and `flex-windows-msvc` if that chain runs and
  succeeds), kept 90 days, holds
  `libtensorflowlite_flex-win.dll`, `libtensorflowlite_flex-win.dll.sha256` and
  `build-info.txt` (TF tag and commit, bazel version, runner image, exact command, size and
  sha256). The `verify` job uploads `flex-windows-<compiler>-verify` and writes the report to
  the run summary. Nothing is uploaded to a release; publishing is a separate, manual step
  (Step 8).

The steps below are the same build done by hand, in Git Bash.

## Prerequisites

1. **Visual Studio 2022** (Build Tools are enough) with "Desktop development with C++".
2. **Bazelisk**, which reads `.bazelversion` from the TF source (TF v2.21.0 pins
   **Bazel 7.7.0**): download `bazelisk-windows-amd64.exe` from
   https://github.com/bazelbuild/bazelisk/releases and put it on PATH as `bazelisk.exe`.
3. **Python 3.11** with `numpy` (`pip install numpy`).
4. **MSYS2** at `C:\msys64`, with the tools bazel's genrules and patches use:
   `C:\msys64\usr\bin\pacman -S --noconfirm --needed patch unzip zip diffutils`.
5. **Git for Windows** with long paths on: `git config --global core.longpaths true`.
6. A **Windows 10/11 SDK older than 10.0.26100** (10.0.22621 works), installed with Visual
   Studio. Newer SDK headers break clang-cl (see Troubleshooting).
7. For the clang-cl variant only: **LLVM 19** (19.1.7 from the llvm-project releases). VS
   2022's STL needs Clang 19 or newer.
8. About **50 GiB of free disk space** on one drive.

## Step 1: Clone TensorFlow

Use a short path:

```bash
git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git /c/fw/tf
cd /c/fw/tf
cat .bazelversion   # 7.7.0
```

## Step 2: Configure the build

```bash
export TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_TENSORRT=0
export TF_SET_ANDROID_WORKSPACE=0 TF_CONFIGURE_IOS=0
export CC_OPT_FLAGS="-Wno-sign-compare"
export USE_DEFAULT_PYTHON_LIB_PATH=1
export PYTHON_BIN_PATH="$(cygpath -m "$(command -v python)")"
export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
export TF_NEED_CLANG=0          # clang-cl variant: TF_NEED_CLANG=1 and
                                # CLANG_COMPILER_PATH="C:/llvm19/bin/clang.exe"
yes "" | python configure.py
```

`TF_OVERRIDE_EIGEN_STRONG_INLINE=1` keeps MSVC compile times reasonable.

Use the clang-cl variant. The MSVC (cl.exe) variant does not build TF v2.21.0 (it fails in
LLVM, see above). For trying it on a later TF, these are the two source patches it needed at
v2.21.0, because TF upstream builds Windows with clang-cl only
(`scripts/flex_desktop/prepare_tf.sh` applies both):

```bash
# XLA's tsl.bzl passes GCC-style Windows copts (-Wno-sign-compare ...), which cl.exe rejects
# with D8021. Ask for the MSVC-style list it already has.
sed -i 's/get_win_copts(is_external, is_msvc = False)/get_win_copts(is_external, is_msvc = True)/' \
  third_party/xla/xla/tsl/tsl.bzl
# zstd's huf_decompress_amd64.S needs C preprocessing; MSVC hands it to ml64.exe (A2008).
# zstd only uses it on ELF and Mach-O targets, so drop it.
sed -i 's|glob(\["decompress/\*_amd64.S"\])|[]|' third_party/xla/third_party/net_zstd.BUILD
```

## Step 3: Create the C wrapper

Copy `scripts/flex_desktop/flex_delegate_plugin.cc` from this repo to
`tensorflow/lite/delegates/flex/flex_delegate_plugin.cc`. It is the same file as on Linux (see
`build_flex_delegate_linux.md`, Step 3): it uses `TFL_EXTERNAL_DELEGATE_EXPORT` from
`external_delegate_interface.h`, and destroys the delegate with
`tflite::TfLiteDelegateFactory::DeleteSimpleDelegate`, the deleter `FlexDelegate::Create()`
pairs with.

The 2.20.0 recipe used `delete reinterpret_cast<tflite::FlexDelegate*>(delegate)`. That skips
the custom deleter and is wrong; the shared wrapper fixes it.

## Step 4: Create the BUILD target

```bash
sed 's/@LIB_NAME@/libtensorflowlite_flex-win.dll/g' \
  /path/to/flutter_litert/scripts/flex_desktop/flex_plugin_target.BUILD.in \
  >> tensorflow/lite/delegates/flex/BUILD
```

The target (same as Linux apart from the name) sets
`local_defines = ["TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY"]`. Without it,
`TFL_EXTERNAL_DELEGATE_EXPORT` expands to `__declspec(dllimport)` and the DLL exports nothing.

## Step 5: Build

```bash
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
export BAZEL_SH=C:/msys64/usr/bin/bash.exe
export BAZEL_VC="$(cygpath -m "$("/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" -latest -products '*' -property installationPath)")/VC"
export BAZEL_LLVM="C:/llvm19"
export BAZEL_WINSDK_FULL_VERSION=10.0.22621.0   # any installed SDK older than 10.0.26100

bazelisk --output_user_root=C:/fw/b build \
  -c opt \
  --config=monolithic \
  --define=tflite_convert_with_select_tf_ops=true \
  --define=with_select_tf_ops=true \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  --local_ram_resources=HOST_RAM*.75 \
  --disk_cache=C:/fw/c/disk \
  --repository_cache=C:/fw/c/repo \
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex-win.dll
```

For the clang-cl variant add:

```bash
  --compiler=clang-cl \
  --extra_toolchains=@local_config_cc//:cc-toolchain-x64_windows-clang-cl \
  --extra_execution_platforms=//tensorflow/tools/toolchains/win:x64_windows-clang-cl \
  --linkopt=/FORCE:MULTIPLE --host_linkopt=/FORCE:MULTIPLE
```

For the MSVC variant add instead (protobuf refuses MSVC + Bazel without
`protobuf_allow_msvc`, and pthreadpool includes `<stdatomic.h>`, which MSVC only allows in C11
mode with its experimental C11 atomics):

```bash
  --define=protobuf_allow_msvc=true \
  --conlyopt=/std:c17 --conlyopt=/experimental:c11atomics \
  --host_conlyopt=/std:c17 --host_conlyopt=/experimental:c11atomics
```

The clang-cl flags are TF's `--config=win_clang` without the `win_clang_base` switches that build protobuf
and absl as separate DLLs: the flex DLL must stay one self-contained file.

Notes:
- `--config=monolithic` links everything into a single DLL.
- `-c opt` enables optimizations; `--config=opt` is not used.
- `--disk_cache` and `--repository_cache` are optional by hand; the CI stages need them.

The output is at `bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex-win.dll`.
The 2.20.0 DLL was about 238 MB.

## Step 6: Verify the DLL

`scripts/flex_desktop/verify_windows.sh <new.dll> <old.dll> <out_dir>` (Git Bash with the MSVC
developer environment, for example after `vcvars64.bat`) does all of this and compares against
the file it replaces (`flex-v1.0.0`, downloaded and only inspected):

```bash
dumpbin /exports libtensorflowlite_flex-win.dll | grep tflite_plugin     # both present
dumpbin /headers libtensorflowlite_flex-win.dll | grep -E 'machine|operating system version|subsystem version'
dumpbin /dependents libtensorflowlite_flex-win.dll                       # system DLLs only
grep -a -oE '2\.21\.[0-9]+' libtensorflowlite_flex-win.dll | sort | uniq -c
```

`machine` must be x64 (8664), the minimum OS and subsystem versions must match the old DLL,
and `/dependents` must list only Windows and MSVC runtime DLLs (no protobuf, absl or TF DLLs).
The script also says whether the DLL imports `vcruntime140_threads.dll` and lists any other C
runtime DLL beyond `msvcp140`, `vcruntime140`, `vcruntime140_1` and `api-ms-win-crt-*`. The
v2.21.0 clang-cl DLL imports `VCRUNTIME140_THREADS.dll` (C11 threads), which the 2.20.0 DLL
did not; it ships only with the VC++ 2015-2022 redistributable 14.38 (VS 2022 17.8) or newer,
so a machine with an older redistributable cannot load the DLL.
The script also writes the added and removed exports compared with the old DLL, and compiles
`smoke_test.c` with `cl` to load the DLL, create and destroy one delegate, and print
`SMOKE OK`.

## Step 7: Test

In the flutter_litert repo on Windows, with the DLL in `%LOCALAPPDATA%\flutter_litert\cache\`:

```powershell
flutter test test\native\flex_delegate_test.dart
```

## Step 8: Publish (manual, not done by the workflow)

The workflow never creates or changes a release. To ship a new build, upload the artifact's DLL
to a new `flex-vX.Y.Z` release in this repo, then point `flutter_litert_flex`'s Windows CMake
download URL and its SHA-256 at it.

## Troubleshooting

**`Checksum was 00b1077e... but wanted 3f986184...` while fetching llvm-project**: see the same
entry in `build_flex_delegate_linux.md` (`scripts/flex_desktop/seed_repo_cache.sh`).

**`declaration of '_m_prefetchw' has a different language linkage`** (clang-cl, in
`prfchwintrin.h`, previous declaration in `winnt.h` of SDK 10.0.26100): the 10.0.26100 SDK
headers clash with Clang 19 and 20. Build against an older SDK by setting
`BAZEL_WINSDK_FULL_VERSION` (for example `10.0.22621.0`) before the first bazel command (or
after `bazel clean --expunge`, since the toolchain is configured once).

**`cl : Command line error D8021 : invalid numeric argument '/Wno-sign-compare'`** or
**`ml64.exe ... huf_decompress_amd64.S ... error A2008`** (MSVC): apply the two MSVC patches in
Step 2.

**Bazel version mismatch**: use Bazelisk; it reads `.bazelversion` (7.7.0 for TF v2.21.0), or
set `USE_BAZEL_VERSION=7.7.0`.

**Missing MSYS2 tools**: set `BAZEL_SH=C:/msys64/usr/bin/bash.exe` and install `patch`,
`unzip`, `zip` and `diffutils` with pacman.

**MSVC not found**: set `BAZEL_VC` to the VS installation's `VC` directory (see Step 5).

**Path too long**: build from a short path (`C:\fw`) and set `--output_user_root` to one.

**Out of memory**: lower `--local_ram_resources` or add `--jobs=4`.

**Exports missing from the DLL**: the target needs
`local_defines = ["TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY"]` (Step 4).

**`configure.py` fails**: make sure `numpy` is installed and `PYTHON_BIN_PATH` points at it.
