#!/usr/bin/env bash
# Clone TensorFlow, add the flutter_litert flex plugin wrapper and BUILD
# target, and run configure.py non-interactively.
#
# Environment:
#   TF_VERSION   tag to clone, for example v2.21.0
#   TF_DIR       where to clone (must not exist yet, or already be a clone)
#   LIB_NAME     libtensorflowlite_flex-linux.so or libtensorflowlite_flex-win.dll
#   FLEX_OS      linux or windows
#   FLEX_VARIANT linux: hermetic. windows: msvc or clang
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$TF_DIR/.git" ]; then
  git clone --depth 1 --branch "$TF_VERSION" \
    https://github.com/tensorflow/tensorflow.git "$TF_DIR"
fi
cd "$TF_DIR"
echo "TensorFlow $(git describe --tags --always) at $(git rev-parse HEAD)"
echo ".bazelversion: $(cat .bazelversion)"

flex_dir=tensorflow/lite/delegates/flex
cp "$here/flex_delegate_plugin.cc" "$flex_dir/flex_delegate_plugin.cc"
if ! grep -q "name = \"$LIB_NAME\"" "$flex_dir/BUILD"; then
  sed "s/@LIB_NAME@/$LIB_NAME/g" "$here/flex_plugin_target.BUILD.in" >> "$flex_dir/BUILD"
fi
tail -n 25 "$flex_dir/BUILD"

# configure.py: no CUDA or ROCm. CC_OPT_FLAGS only applies under --config=opt,
# which the build does not use; it is set so nothing like -march=native can
# slip in. Any prompt not covered by an env var takes its default.
export TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_TENSORRT=0
export TF_SET_ANDROID_WORKSPACE=0 TF_CONFIGURE_IOS=0
export CC_OPT_FLAGS="-Wno-sign-compare"
export USE_DEFAULT_PYTHON_LIB_PATH=1
if [ "$FLEX_OS" = "windows" ]; then
  PYTHON_BIN_PATH="$(cygpath -m "$(command -v python)")"
  export PYTHON_BIN_PATH
  export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
  if [ "$FLEX_VARIANT" = "clang" ]; then
    export TF_NEED_CLANG=1
    export CLANG_COMPILER_PATH="${BAZEL_LLVM:-C:/Program Files/LLVM}/bin/clang.exe"
  else
    export TF_NEED_CLANG=0
    # XLA's copy of tsl.bzl (in TF's tree at third_party/xla) passes its
    # Windows copts in GCC style (-Wno-sign-compare and so on), which only
    # clang-cl accepts. Ask for the MSVC-style list it already has, the same
    # flags TF's own tensorflow.bzl uses on Windows.
    tsl_bzl=third_party/xla/xla/tsl/tsl.bzl
    if grep -q 'get_win_copts(is_external, is_msvc = False)' "$tsl_bzl"; then
      sed -i 's/get_win_copts(is_external, is_msvc = False)/get_win_copts(is_external, is_msvc = True)/' "$tsl_bzl"
      echo "patched $tsl_bzl for MSVC:"
      grep -n 'get_win_copts(is_external' "$tsl_bzl"
    fi
    # zstd's x86_64 assembly (decompress/huf_decompress_amd64.S) is GNU
    # assembler syntax that needs C preprocessing; MSVC hands .S files to
    # ml64.exe, which rejects it. zstd only uses that code on ELF and Mach-O
    # targets, so on Windows the file is dead weight: drop it.
    zstd_build=third_party/xla/third_party/net_zstd.BUILD
    if grep -q 'glob(\["decompress/\*_amd64.S"\])' "$zstd_build"; then
      sed -i 's|glob(\["decompress/\*_amd64.S"\])|[]|' "$zstd_build"
      echo "patched $zstd_build for MSVC:"
      grep -n -A2 'x86_64": ' "$zstd_build"
    fi
  fi
else
  PYTHON_BIN_PATH="$(command -v python3)"
  export PYTHON_BIN_PATH
  # Linux builds use TF's hermetic clang toolchain (USE_HERMETIC_CC_TOOLCHAIN=1
  # in TF's .bazelrc); a local clang path is still required by configure.py.
  export TF_NEED_CLANG=1
  clang_path=""
  for c in /usr/lib/llvm-18/bin/clang /usr/lib/llvm-17/bin/clang "$(command -v clang || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then clang_path="$c"; break; fi
  done
  export CLANG_COMPILER_PATH="$clang_path"
fi
(yes "" || true) | "$PYTHON_BIN_PATH" configure.py
echo "---- .tf_configure.bazelrc ----"
cat .tf_configure.bazelrc
