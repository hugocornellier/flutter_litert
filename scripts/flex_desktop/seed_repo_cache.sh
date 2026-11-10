#!/usr/bin/env bash
# Put source archives whose upstream URLs no longer serve the bytes TF pins
# into bazel's repository cache, so bazel finds them by sha256 and never
# downloads them. Each archive is only accepted when its sha256 is exactly the
# one TF pins, so this cannot change what is built.
#
# Why: for TF v2.21.0, github.com now serves the llvm-project archive at
# 909041e4 with sha256 00b1077e..., not the 3f986184... XLA pins, and the TF
# mirror (storage.googleapis.com/mirror.tensorflow.org) returns 404, so every
# bazel build of v2.21.0 stops while fetching @llvm-raw.
#
#   seed_repo_cache.sh <repository_cache_dir>
set -uo pipefail

repo_cache="$1"
if command -v cygpath >/dev/null 2>&1; then repo_cache="$(cygpath -u "$repo_cache")"; fi

# sha256 then candidate URLs, one archive per line.
entries=(
  "3f986184ee126677dbd77edb16d6b82c057ec869fefd7a9871979941e52e837a https://mirror.bazel.build/github.com/llvm/llvm-project/archive/909041e4802c4b9a2223ca04099f35bf1dbbd460.tar.gz https://github.com/llvm/llvm-project/archive/909041e4802c4b9a2223ca04099f35bf1dbbd460.tar.gz"
)

for entry in "${entries[@]}"; do
  read -r -a parts <<< "$entry"
  want="${parts[0]}"
  dest="$repo_cache/content_addressable/sha256/$want"
  if [ -f "$dest/file" ]; then
    echo "repository cache already has $want"
    continue
  fi
  mkdir -p "$dest"
  ok=0
  for url in "${parts[@]:1}"; do
    tmp="$dest/file.part"
    if ! curl -fsSL --retry 3 -o "$tmp" "$url"; then
      echo "::warning::download failed: $url"
      rm -f "$tmp"
      continue
    fi
    got="$(sha256sum "$tmp" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
      mv "$tmp" "$dest/file"
      echo "seeded $want from $url"
      ok=1
      break
    fi
    echo "::warning::$url has sha256 $got, not $want"
    rm -f "$tmp"
  done
  if [ "$ok" -eq 0 ]; then
    rmdir "$dest" 2>/dev/null || true
    echo "::warning::no source served the pinned bytes for $want; bazel will try upstream itself"
  fi
done
