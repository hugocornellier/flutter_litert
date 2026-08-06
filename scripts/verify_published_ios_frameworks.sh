#!/usr/bin/env bash
# Verify that the iOS Core ML frameworks CONSUMERS ACTUALLY DOWNLOAD carry the
# NPU entry points, for both packaging channels.
#
# Why this exists
# ---------------
# The regular iOS CI job builds against ios/TensorFlowLiteCCoreML.xcframework
# from the working copy (scripts/use_local_coreml_xcframework.sh redirects the
# SPM binary target at it). That validates the framework on disk and says
# nothing about the artifacts a consumer fetches, so two separate shipping bugs
# reached users unnoticed:
#
#   1. Package.swift pinned a Core ML release predating the NPU entry points, so
#      SPM consumers got a framework without them.
#   2. The podspec downloaded libs-v0.1.8, whose Core ML framework never
#      contained them, so CocoaPods consumers got the same failure after (1) was
#      fixed for SPM only.
#
# Both failed identically and silently: `pod install`/`swift build` succeeded,
# the app built and ran, and accelerator registration returned
# kLiteRtStatusErrorUnsupported for every model at runtime.
#
# This script closes that gap by resolving the URLs from the podspec and
# Package.swift themselves (never hardcoded here, so a URL bump cannot drift
# past the check) and asserting the downloaded bytes are correct.
#
# FlutterTfLiteCoreMlNpuDelegateCreate is added by
# patches/litert_coreml_npu_ios.patch and exists in no upstream TensorFlow build
# at any version, so its absence is an unambiguous signal that a channel is
# serving a stale or unpatched artifact.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PODSPEC="$REPO_ROOT/ios/flutter_litert.podspec"
PACKAGE_SWIFT="$REPO_ROOT/ios/flutter_litert/Package.swift"
NPU_SYMBOL="FlutterTfLiteCoreMlNpuDelegateCreate"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

failures=0
fail() { echo "  FAIL: $*"; failures=$((failures + 1)); }

# Exported Mach-O symbol names live in the binary's string table, so grep finds
# them without needing macOS-only tooling; this keeps the job off the contended
# macOS runners. The symbol name is unique to our patch, so a match is not
# ambiguous.
assert_npu_symbols() {
  local xcframework_dir="$1" channel="$2" found_any=0
  for slice in ios-arm64 ios-arm64_x86_64-simulator; do
    local binary="$xcframework_dir/$slice/TensorFlowLiteCCoreML.framework/TensorFlowLiteCCoreML"
    if [ ! -f "$binary" ]; then
      fail "$channel: missing slice $slice"
      continue
    fi
    if grep -aqc "$NPU_SYMBOL" "$binary" 2>/dev/null; then
      echo "  ok: $channel $slice exports $NPU_SYMBOL"
      found_any=1
    else
      fail "$channel: $slice does NOT export $NPU_SYMBOL (unpatched artifact)"
    fi
  done
  [ "$found_any" -eq 1 ] || fail "$channel: no usable slice found"
}

echo "== CocoaPods channel =="
# The podspec downloads one bundle guarded by a marker; pull the URL it uses.
pods_url="$(grep -oE "https://github\.com/[^']*/ios-frameworks\.zip" "$PODSPEC" | head -1)"
if [ -z "$pods_url" ]; then
  fail "could not resolve the ios-frameworks.zip URL from $PODSPEC"
else
  echo "  url: $pods_url"
  curl -fsSL -o "$WORK/pods.zip" "$pods_url" || fail "download failed: $pods_url"
  if [ -s "$WORK/pods.zip" ]; then
    mkdir -p "$WORK/pods" && unzip -qo "$WORK/pods.zip" -d "$WORK/pods"
    assert_npu_symbols "$WORK/pods/TensorFlowLiteCCoreML.xcframework" "cocoapods"
  fi
fi

echo
echo "== SwiftPM channel =="
# Verify EVERY binary target, not just Core ML. A stale pin on any of them is
# the same class of bug: the manifest and the published bytes drift apart, and
# nothing notices until a consumer hits it. Targets are read from the manifest
# so a newly added one is covered automatically rather than silently skipped.
targets="$(python3 - "$PACKAGE_SWIFT" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
# .binaryTarget(name: "X", url: "...", checksum: "...")
for m in re.finditer(
    r'\.binaryTarget\(\s*name:\s*"([^"]+)"\s*,\s*url:\s*"([^"]+)"\s*,\s*checksum:\s*"([a-f0-9]{64})"',
    src):
    print(f"{m.group(1)}\t{m.group(2)}\t{m.group(3)}")
PY
)"

if [ -z "$targets" ]; then
  fail "could not parse any binaryTarget from $PACKAGE_SWIFT"
else
  echo "  found $(printf '%s\n' "$targets" | wc -l | tr -d ' ') binary targets"
  while IFS=$'\t' read -r name url checksum; do
    [ -n "$name" ] || continue
    out="$WORK/$name.zip"
    if ! curl -fsSL -o "$out" "$url"; then
      fail "swiftpm $name: download failed ($url)"
      continue
    fi
    # A checksum mismatch means SPM refuses to resolve the package at all, so
    # this is a hard consumer-facing break rather than a warning.
    actual="$(shasum -a 256 "$out" | cut -d' ' -f1)"
    if [ "$actual" = "$checksum" ]; then
      echo "  ok: $name checksum matches the pin"
    else
      fail "swiftpm $name: checksum mismatch (pinned $checksum, got $actual)"
    fi
    if [ "$name" = "TensorFlowLiteCCoreML" ]; then
      mkdir -p "$WORK/spm" && unzip -qo "$out" -d "$WORK/spm"
      assert_npu_symbols "$WORK/spm/TensorFlowLiteCCoreML.xcframework" "swiftpm"
    fi
  done <<< "$targets"
fi

echo
if [ "$failures" -ne 0 ]; then
  echo "FAILED: $failures problem(s) in the published iOS frameworks."
  echo "A consumer installing this version would not get working Core ML NPU."
  exit 1
fi
echo "PASSED: both channels serve a Core ML framework with the NPU entry points."
