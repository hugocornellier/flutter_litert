#!/usr/bin/env bash
# Free runner disk and choose where the TensorFlow checkout, bazel's
# output_user_root and the stage caches go. Appends the chosen paths to
# $GITHUB_ENV: TF_DIR, OUTPUT_USER_ROOT, CACHE_ROOT, DISK_CACHE, REPO_CACHE,
# ARCHIVE_DIR, DIST_DIR.
#
#   layout.sh linux|windows
set -euo pipefail

os="$1"
free_kb() { df -Pk "$1" | awk 'NR == 2 { print $4 }'; }

if [ "$os" = "linux" ]; then
  echo "Before cleanup:"; df -h / /mnt 2>/dev/null || df -h /
  # Preinstalled toolchains this build never uses. Bazel brings its own JDK,
  # and TF's hermetic toolchain brings its own clang and Python.
  sudo rm -rf \
    /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
    /opt/hostedtoolcache/CodeQL /opt/hostedtoolcache/go /opt/hostedtoolcache/PyPy \
    /opt/hostedtoolcache/Ruby /opt/hostedtoolcache/node /usr/share/swift \
    /usr/local/share/boost /usr/local/share/powershell /usr/local/share/chromium \
    /usr/local/julia* /usr/share/miniconda /opt/az /opt/microsoft /opt/google \
    /usr/lib/firefox /opt/pipx "$AGENT_TOOLSDIRECTORY/CodeQL" 2>/dev/null || true
  sudo docker image prune --all --force >/dev/null 2>&1 || true
  sudo apt-get clean || true
  echo "After cleanup:"; df -h / /mnt 2>/dev/null || df -h /

  candidates=("$HOME/flexwork")
  if mountpoint -q /mnt; then
    sudo mkdir -p /mnt/flexwork
    sudo chown "$(id -u):$(id -g)" /mnt/flexwork
    candidates+=(/mnt/flexwork)
  fi
  mkdir -p "$HOME/flexwork"
else
  echo "Drives:"; df -h 2>/dev/null | grep -E '^[A-Z]:|/[a-z]$' || df -h
  # Real-time scanning of hundreds of thousands of bazel outputs slows the
  # build a lot.
  powershell -NoProfile -Command "Set-MpPreference -DisableRealtimeMonitoring \$true" || true
  candidates=()
  for d in /c /d; do
    if [ -d "$d" ]; then mkdir -p "$d/fw"; candidates+=("$d/fw"); fi
  done
fi

big=""; small=""; big_kb=0; small_kb=0
for c in "${candidates[@]}"; do
  kb=$(free_kb "$c")
  echo "$c: $((kb / 1024 / 1024)) GiB free"
  if [ "$kb" -gt "$big_kb" ]; then
    small="$big"; small_kb=$big_kb; big="$c"; big_kb=$kb
  elif [ "$kb" -gt "$small_kb" ]; then
    small="$c"; small_kb=$kb
  fi
done

# Caches go on the second volume only when it has room for them (40 GiB).
cache_vol="$big"
if [ -n "$small" ] && [ "$small_kb" -gt $((40 * 1024 * 1024)) ]; then cache_vol="$small"; fi

to_native() { if [ "$os" = "windows" ]; then cygpath -m "$1"; else echo "$1"; fi; }

if [ "$os" = "linux" ]; then
  # Extra swap so a few large TF translation units in parallel do not get
  # the build OOM-killed on a 16 GiB runner.
  sudo fallocate -l 8G "$big/swapfile2" && sudo chmod 600 "$big/swapfile2" \
    && sudo mkswap "$big/swapfile2" >/dev/null && sudo swapon "$big/swapfile2" || true
  free -h
fi

{
  echo "TF_DIR=$(to_native "$big")/tf"
  echo "OUTPUT_USER_ROOT=$(to_native "$big")/b"
  echo "CACHE_ROOT=$(to_native "$cache_vol")/c"
  echo "DISK_CACHE=$(to_native "$cache_vol")/c/disk"
  echo "REPO_CACHE=$(to_native "$cache_vol")/c/repo"
  echo "ARCHIVE_DIR=$(to_native "$big")/archive"
  echo "DIST_DIR=$(to_native "$big")/dist"
} | tee -a "$GITHUB_ENV"
