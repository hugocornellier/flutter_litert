#!/bin/sh
# Re-pin the LLVM source archive in a TensorFlow v2.21.0 checkout.
#
# Usage: pin_tf_llvm_archive.sh <tf-src>
#
# TF v2.21.0 (third_party/xla/third_party/llvm/workspace.bzl) fetches
# llvm-project at commit 909041e4802c4b9a2223ca04099f35bf1dbbd460 with sha256
# 3f986184...837a. The mirror.tensorflow.org copy now returns 404, and the
# github.com archive of the same commit is served with different bytes
# (sha256 00b1077e...6248), so every bazel build of TF 2.21.0 fails in
# repository fetch with "Checksum was 00b1077e... but wanted 3f986184...".
#
# The commit id still pins the source. This script swaps the expected sha256
# for the one github.com serves today, and pins THAT value here so bazel still
# refuses any other bytes. It fails if the checkout does not carry the
# expected old pin (for example on a different TF version).
set -eu

tf="$1"
commit=909041e4802c4b9a2223ca04099f35bf1dbbd460
old=3f986184ee126677dbd77edb16d6b82c057ec869fefd7a9871979941e52e837a
new=00b1077e029fa57e6f2d9ac24936a49acf23ebc051b04f487131116258be6248

files=$(grep -rl "$old" "$tf/third_party" "$tf/tensorflow/workspace2.bzl" 2>/dev/null || true)
if [ -z "$files" ]; then
  echo "no file pins llvm sha256 $old; nothing to do (different TF version?)"
  grep -rn "LLVM_COMMIT =\|LLVM_SHA256 =" "$tf/third_party" | head
  exit 1
fi
for f in $files; do
  grep -q "$commit" "$f" || { echo "$f pins $old but not commit $commit"; exit 1; }
  sed -i.bak "s/$old/$new/" "$f"
  rm -f "$f.bak"
  echo "re-pinned llvm $commit archive sha256 in $f"
  grep -n "LLVM_COMMIT =\|LLVM_SHA256 =" "$f"
done
