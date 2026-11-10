#!/bin/sh
# Make TF v2.21.0's apple_support toolchain pass its flags and Xcode env to
# Objective-C(++) compiles.
#
# Usage: patch_tf_apple_support.sh <tf-src>
#
# TF v2.21.0 moved to apple_support 1.24.5, whose crosstool scopes "-c <src>",
# "-target ..." and the XCODE_VERSION_OVERRIDE / APPLE_SDK_* env to
# ACTION_NAME_GROUPS.all_cc_compile_actions. Only rules_cc 0.2+ puts
# objc-compile and objc++-compile in that group, and TF's WORKSPACE resolves an
# older rules_cc, so the first .mm compile (gpu/metal/common.mm,
# coreml/coreml_executor.mm) fails with "Error: DEVELOPER_DIR not set." from
# wrapped_clang. The same goes for objc-executable, the ObjcLink of
# macos_dylib, which rules_cc 0.2+ lists in cc_link_executable_actions.
# 2.20.0 used apple_support 1.18.1 and did not hit this.
#
# This copies patches/apple_support_objc_compile_actions.patch into the
# checkout and adds it as patch_file to the build_bazel_apple_support
# tf_http_archive (sha256 still checked), so bazel patches apple_support at
# fetch time. It fails if the checkout does not carry apple_support 1.24.5.
set -eu

tf="$1"
here="$(cd "$(dirname "$0")/.." && pwd)"
patch_name=apple_support_objc_compile_actions.patch
ws="$tf/tensorflow/workspace2.bzl"
sha=1ae6fcf983cff3edab717636f91ad0efff2e5ba75607fdddddfd6ad0dbdfaf10

grep -q "apple_support.1.24.5.tar.gz" "$ws" || { echo "$ws does not fetch apple_support 1.24.5"; exit 1; }
grep -q "sha256 = \"$sha\"" "$ws" || { echo "$ws does not pin apple_support sha256 $sha"; exit 1; }
if grep -q "$patch_name" "$ws"; then
  echo "already patched"
  exit 0
fi

cp "$here/patches/$patch_name" "$tf/third_party/$patch_name"
# Insert the patch_file line right after apple_support's sha256 line.
awk -v sha="$sha" -v label="//third_party:$patch_name" '
  { print }
  index($0, "sha256 = \"" sha "\"") { match($0, /^ */); printf "%spatch_file = [\"%s\"],\n", substr($0, 1, RLENGTH), label; n++ }
  END { if (n != 1) exit 1 }
' "$ws" > "$ws.new"
mv "$ws.new" "$ws"
echo "added $patch_name to build_bazel_apple_support in $ws"
grep -n -B2 -A3 "$patch_name" "$ws"
