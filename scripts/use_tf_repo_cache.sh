#!/bin/bash
# Makes every bazel command in a TensorFlow checkout find the archives that
# scripts/flex/seed_bazel_repo_cache.sh puts in a repository cache, so bazel
# never downloads them from URLs that no longer serve the pinned bytes.
#
# github.com has served the llvm-project 909041e4 archive TF v2.21.0 pins with
# two different sha256 values: 00b1077e... from about 2026-11-09, and the
# pinned 3f986184... again on 2026-11-13, which failed every Apple build that
# had swapped the pin for 00b1077e.... The seeded cache holds the pinned bytes
# from mirror.bazel.build, so the pin no longer depends on github.com.
#
# Usage: use_tf_repo_cache.sh <tf-src> [repository_cache_dir]
set -euo pipefail

tf="$1"
cache="${2:-$HOME/bazel-repo-cache}"
bash "$(dirname "$0")/flex/seed_bazel_repo_cache.sh" "$cache"
# TF's .bazelrc imports .bazelrc.user from the workspace.
echo "common --repository_cache=$cache" >> "$tf/.bazelrc.user"
echo "bazel in $tf uses the repository cache $cache"
