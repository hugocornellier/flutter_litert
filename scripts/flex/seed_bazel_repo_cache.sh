#!/bin/bash
# Puts archives that TF v2.21.0 pins but can no longer fetch from its own URLs
# into a Bazel repository cache, so `bazel build --repository_cache=<dir>`
# finds them by sha256 and never downloads them.
#
# llvm-project 909041e4 (pinned by XLA at sha256 3f986184...): the TF mirror
# (storage.googleapis.com/mirror.tensorflow.org) returns 404, and github.com
# has served that archive with sha256 00b1077e... as well as the pinned bytes,
# so a bazel build of v2.21.0 that downloads it can stop while fetching
# @llvm-raw. mirror.bazel.build has the original bytes; github.com is the
# fallback.
#
# A file is only kept when its sha256 is the pinned one.
#
# Usage: seed_bazel_repo_cache.sh <repository_cache_dir>
set -euo pipefail

CACHE="$1"
mkdir -p "$CACHE"

# sha256 url [url...]
ARCHIVES=(
  "3f986184ee126677dbd77edb16d6b82c057ec869fefd7a9871979941e52e837a https://mirror.bazel.build/github.com/llvm/llvm-project/archive/909041e4802c4b9a2223ca04099f35bf1dbbd460.tar.gz https://github.com/llvm/llvm-project/archive/909041e4802c4b9a2223ca04099f35bf1dbbd460.tar.gz"
)

for entry in "${ARCHIVES[@]}"; do
  read -r sha urls <<< "$entry"
  dest="$CACHE/content_addressable/sha256/$sha"
  if [ -f "$dest/file" ]; then
    echo "already cached: $sha"
    continue
  fi
  mkdir -p "$dest"
  ok=0
  for url in $urls; do
    echo "fetching $url"
    if curl -fsSL --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
        --retry 3 --retry-delay 5 --retry-all-errors -o "$dest/file.part" "$url"; then
      got="$(shasum -a 256 "$dest/file.part" | cut -d' ' -f1)"
      if [ "$got" = "$sha" ]; then
        mv "$dest/file.part" "$dest/file"
        echo "cached $sha ($(wc -c < "$dest/file") bytes)"
        ok=1
        break
      fi
      echo "::warning::$url has sha256 $got, wanted $sha"
    fi
    rm -f "$dest/file.part"
  done
  if [ "$ok" != 1 ]; then
    echo "::error::could not get an archive with sha256 $sha"
    exit 1
  fi
done
