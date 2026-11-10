#!/bin/bash
# Frees disk space on a GitHub-hosted macOS runner before a large bazel build
# (TensorFlow flex delegate). Best effort: every removal may fail without
# failing the job. Keeps the selected Xcode (and its iOS/macOS SDKs), Homebrew
# and the Python tool cache that actions/setup-python uses.
#
# Removes: all simulator runtimes and devices (a bazel build for the iOS
# simulator needs the SDK, not a runtime), Android SDK and NDKs,
# .NET, PowerShell, Haskell, Rust, Chromium, Mono/Xamarin leftovers, and the
# non-Python language tool caches.
set -u

echo "== Disk before cleanup"
df -h /

echo "Selected Xcode: $(xcode-select -p)"
xcodebuild -version || true

# The other Xcodes are kept by default: Bazel's Xcode discovery reads them,
# and the simulator runtimes below free more than enough. Set
# REMOVE_OTHER_XCODES=1 to delete them too (about 50 GB more).
if [ "${REMOVE_OTHER_XCODES:-0}" = 1 ]; then
  selected_real="$(cd "$(xcode-select -p)/../.." && pwd -P)"
  for x in /Applications/Xcode*.app; do
    [ -e "$x" ] || continue
    real="$(cd "$x" && pwd -P)"
    if [ "$real" = "$selected_real" ]; then
      echo "keep $x"
      continue
    fi
    echo "remove $x"
    sudo rm -rf "$x" || true
  done
fi

echo "== Simulator runtimes before"
xcrun simctl runtime list 2>/dev/null | head -40 || true
xcrun simctl shutdown all 2>/dev/null || true
xcrun simctl delete all 2>/dev/null || true
xcrun simctl runtime delete all 2>/dev/null || true
sudo rm -rf ~/Library/Developer/CoreSimulator/Caches 2>/dev/null || true
echo "== Simulator runtimes after"
xcrun simctl runtime list 2>/dev/null | head -20 || true

for d in \
  "${ANDROID_HOME:-/nonexistent}" \
  "${ANDROID_SDK_ROOT:-/nonexistent}" \
  "$HOME/Library/Android" \
  /usr/local/lib/android \
  /usr/local/share/dotnet "$HOME/.dotnet" \
  /usr/local/share/powershell \
  /usr/local/.ghcup "$HOME/.ghcup" \
  "$HOME/.rustup" "$HOME/.cargo" \
  /usr/local/share/chromium \
  /Library/Frameworks/Mono.framework \
  /Library/Frameworks/Xamarin.iOS.framework \
  /Library/Frameworks/Xamarin.Android.framework \
  "$HOME/Library/Caches/Homebrew" \
  "${RUNNER_TOOL_CACHE:-/nonexistent}/Ruby" \
  "${RUNNER_TOOL_CACHE:-/nonexistent}/go" \
  "${RUNNER_TOOL_CACHE:-/nonexistent}/node" \
  "${RUNNER_TOOL_CACHE:-/nonexistent}/PyPy" \
  "${RUNNER_TOOL_CACHE:-/nonexistent}/CodeQL" \
  "${RUNNER_TOOL_CACHE:-/nonexistent}/Java_Temurin-Hotspot_jdk" \
  ; do
  [ -e "$d" ] || continue
  echo "remove $d"
  sudo rm -rf "$d" || true
done

echo "== Disk after cleanup"
df -h /
exit 0
