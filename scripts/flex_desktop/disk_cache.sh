#!/usr/bin/env bash
# Pack or unpack the bazel caches handed from one build stage to the next.
# <cache_root> holds disk/ (--disk_cache) and repo/ (--repository_cache).
#
#   disk_cache.sh pack   <cache_root> <archive_dir>
#   disk_cache.sh unpack <cache_root> <archive_dir>
#
# The archive is <archive_dir>/disk-cache.tar.zst when zstd is available,
# otherwise <archive_dir>/disk-cache.tar (upload-artifact then compresses it).
set -euo pipefail

mode="$1"
cache_dir="$2"
archive_dir="$3"
if command -v cygpath >/dev/null 2>&1; then
  # Git Bash on Windows: GNU tar reads "D:/x" as a remote host, so use /d/x.
  cache_dir="$(cygpath -u "$cache_dir")"
  archive_dir="$(cygpath -u "$archive_dir")"
fi
mkdir -p "$cache_dir" "$archive_dir"
# Git Bash has no zstd; the workflow installs MSYS2's.
if ! command -v zstd >/dev/null 2>&1 && [ -x /c/msys64/usr/bin/zstd ]; then
  PATH="$PATH:/c/msys64/usr/bin"
fi

case "$mode" in
  pack)
    # Unfinished writes live in tmp/ and are useless to the next stage.
    find "$cache_dir" -maxdepth 2 -type d -name tmp -prune -exec rm -rf {} +
    echo "disk cache: $(du -sh "$cache_dir" | cut -f1), $(find "$cache_dir" -type f | wc -l) files"
    if command -v zstd >/dev/null 2>&1; then
      tar -C "$cache_dir" -cf - . | zstd -T0 -3 -q -o "$archive_dir/disk-cache.tar.zst"
    else
      tar -C "$cache_dir" -cf "$archive_dir/disk-cache.tar" .
    fi
    ls -l "$archive_dir"
    ;;
  unpack)
    if [ -f "$archive_dir/disk-cache.tar.zst" ]; then
      zstd -d -q -c "$archive_dir/disk-cache.tar.zst" | tar -C "$cache_dir" -xf -
      rm -f "$archive_dir/disk-cache.tar.zst"
    elif [ -f "$archive_dir/disk-cache.tar" ]; then
      tar -C "$cache_dir" -xf "$archive_dir/disk-cache.tar"
      rm -f "$archive_dir/disk-cache.tar"
    else
      echo "No disk cache archive in $archive_dir; starting cold."
      exit 0
    fi
    echo "restored disk cache: $(du -sh "$cache_dir" | cut -f1)"
    ;;
  *)
    echo "usage: $0 pack|unpack <disk_cache_dir> <archive_dir>" >&2
    exit 2
    ;;
esac
