#!/bin/sh
# Patch pthreadpool for Apple platforms in a TensorFlow v2.21.0 checkout
# (flutter_litert 4.1.1, XNNPACK delegate speed).
#
# Usage: patch_tf_pthreadpool.sh <tf-src>
#
# TF v2.21.0 pins pthreadpool 0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e, which
# dropped its Grand Central Dispatch backend (src/gcd.c). Apple platforms now
# use the pthreads backend (src/pthreads.c): a worker spins 1000 iterations
# and then sleeps on a condition variable, so it has to be woken up again for
# every parallel section of an inference, and the wait loops hold the pool
# mutex while they spin. With the XNNPACK delegate that made the Interpreter
# about 8% slower on a physical iPhone in flutter_litert 4.1.0 (plain CPU,
# flex and Core ML were unchanged). GCD never slept between the sections of
# one inference. Android keeps its own spin count of 100000.
#
# patches/pthreadpool_apple_spin.patch makes the condition-variable waits in
# src/pthreads.c spin without the mutex (taking it only to sleep, with a
# re-check under the lock), makes the main thread poll the completion flag
# with plain loads instead of an atomic exchange per iteration (the exchange
# starved the signalling worker on the iPhone), and gives Apple the Android
# spin count in src/threadpool-common.h. It applies to that exact pthreadpool
# revision; see patches/README.md. This script registers the patch in both
# places TF fetches pthreadpool from:
#   - tensorflow/workspace2.bzl (bazel, the iOS frameworks and the macOS arm64
#     slice): adds patch_file to the pthreadpool tf_http_archive;
#   - tensorflow/lite/cmake/DownloadPThreadPool.cmake (CMake, the macOS x86_64
#     slice): adds a PATCH_COMMAND to the ExternalProject.
# It fails if the checkout does not pin that pthreadpool revision.
set -eu

tf="$1"
here=$(cd "$(dirname "$0")" && pwd)
patch_src="$here/../patches/pthreadpool_apple_spin.patch"
sha=f602ab141bdc5d5872a79d6551e9063b5bfa7ad6ad60cceaa641de5c45c86d70
commit=0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e
bzl="$tf/tensorflow/workspace2.bzl"
cmake="$tf/tensorflow/lite/cmake/DownloadPThreadPool.cmake"

test -f "$patch_src" || { echo "missing $patch_src"; exit 1; }
for f in "$bzl" "$cmake"; do
  grep -q "$sha" "$f" || { echo "$f does not pin pthreadpool sha256 $sha (different TF version?)"; exit 1; }
  grep -q "$commit" "$f" || { echo "$f pins $sha but not commit $commit"; exit 1; }
done
if grep -q "pthreadpool_apple_spin.patch" "$bzl" "$cmake"; then
  echo "pthreadpool patch already registered"; exit 1
fi

cp "$patch_src" "$tf/third_party/pthreadpool_apple_spin.patch"

# bazel: the pthreadpool entry is the tf_http_archive whose strip_prefix names
# the pinned commit; add patch_file right after that line.
sed -i.bak "/strip_prefix = \"pthreadpool-$commit\",/a\\
        patch_file = [\"//third_party:pthreadpool_apple_spin.patch\"],
" "$bzl"
rm -f "$bzl.bak"
grep -q "patch_file = \[\"//third_party:pthreadpool_apple_spin.patch\"\]" "$bzl" || { echo "workspace2.bzl edit failed"; exit 1; }

# CMake: the file is passed through configure_file, so use an absolute path.
abs=$(cd "$tf" && pwd)/third_party/pthreadpool_apple_spin.patch
sed -i.bak "/^  CONFIGURE_COMMAND \"\"/i\\
  PATCH_COMMAND patch -p1 -i \"$abs\"
" "$cmake"
rm -f "$cmake.bak"
grep -q "PATCH_COMMAND patch -p1 -i" "$cmake" || { echo "DownloadPThreadPool.cmake edit failed"; exit 1; }

echo "registered pthreadpool_apple_spin.patch for pthreadpool $commit"
grep -n -B3 -A2 "pthreadpool_apple_spin" "$bzl"
grep -n -B2 -A1 "PATCH_COMMAND" "$cmake"
