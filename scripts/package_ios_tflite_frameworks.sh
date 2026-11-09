#!/usr/bin/env bash
# Packages the per-slice bazel outputs of TensorFlowLiteC and TensorFlowLiteCMetal
# into the two forms flutter_litert consumes on iOS:
#
# 1. CocoaPods (the podspec downloads ios-frameworks.zip): an xcframework per
#    library with an `ios-arm64` device slice and a universal
#    `ios-arm64_x86_64-simulator` slice, no per-bundle Info.plist. This is the
#    layout of libs-v0.1.8 / libs-v0.1.9. If a TensorFlowLiteCCoreML.xcframework
#    is given, it is added so the zip matches the full ios-frameworks.zip.
# 2. SwiftPM (Package.swift binary targets): <name>-spm.xcframework.zip with an
#    `ios-arm64` slice and an arm64-only `ios-arm64-simulator` slice
#    (lipo -thin arm64 of the universal one), each framework carrying its own
#    Info.plist. This is the layout of the flex-v1.1.1 TensorFlowLiteC-spm and
#    TensorFlowLiteCMetal-spm zips. Both xcframeworks are written by
#    `xcodebuild -create-xcframework` so the slice identifiers in Info.plist
#    match the directories (see the comment in ios/flutter_litert/Package.swift).
#
# Usage:
#   scripts/package_ios_tflite_frameworks.sh <raw-dir> <out-dir> [coreml.xcframework]
#
# <raw-dir> holds one directory per bazel build, named <Name>-<config>, each with
# the unzipped <Name>.framework from //tensorflow/lite/ios:<Name>_framework:
#   TensorFlowLiteC-ios_arm64, TensorFlowLiteC-ios_sim_arm64, TensorFlowLiteC-ios_x86_64,
#   TensorFlowLiteCMetal-ios_arm64, ...
#
# Writes to <out-dir>:
#   pods/<Name>.xcframework, ios-frameworks.zip,
#   spm/<Name>.xcframework, <Name>-spm.xcframework.zip,
#   and a .sha256 file next to each zip.
set -euo pipefail

raw="$(cd "$1" && pwd)"
mkdir -p "$2"
out="$(cd "$2" && pwd)"
coreml="${3:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Per-bundle Info.plist, byte-for-byte the content of the flex-v1.1.1 SPM zips.
write_bundle_plist() {
  local name="$1" platform="$2" dest="$3"
  cat > "$dest" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleDevelopmentRegion</key><string>en</string>
<key>CFBundleExecutable</key><string>${name}</string>
<key>CFBundleIdentifier</key><string>org.tensorflow.${name}</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>${name}</string>
<key>CFBundlePackageType</key><string>FMWK</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleSupportedPlatforms</key><array><string>${platform}</string></array>
<key>MinimumOSVersion</key><string>13.0</string>
</dict></plist>
PLIST
}

# xcframework Info.plist for the CocoaPods form, identical to libs-v0.1.9's.
# Only used if xcodebuild refuses frameworks that have no bundle Info.plist.
write_pods_xcframework_plist() {
  local name="$1" dest="$2"
  cat > "$dest" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AvailableLibraries</key>
	<array>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64</string>
			<key>LibraryPath</key>
			<string>${name}.framework</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
		</dict>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64_x86_64-simulator</string>
			<key>LibraryPath</key>
			<string>${name}.framework</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
				<string>x86_64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
			<key>SupportedPlatformVariant</key>
			<string>simulator</string>
		</dict>
	</array>
	<key>CFBundlePackageType</key>
	<string>XFWK</string>
	<key>XCFrameworkFormatVersion</key>
	<string>1.0</string>
</dict>
</plist>
PLIST
}

sha_file() {
  (cd "$(dirname "$1")" && shasum -a 256 "$(basename "$1")" > "$(basename "$1").sha256")
  cat "$1.sha256"
}

mkdir -p "$out/pods" "$out/spm"
for name in TensorFlowLiteC TensorFlowLiteCMetal; do
  dev="$raw/$name-ios_arm64/$name.framework"
  sim_arm64="$raw/$name-ios_sim_arm64/$name.framework"
  sim_x86="$raw/$name-ios_x86_64/$name.framework"
  for d in "$dev" "$sim_arm64" "$sim_x86"; do
    [ -f "$d/$name" ] || { echo "missing $d/$name"; exit 1; }
  done
  echo "=== $name: bazel framework contents (device) ==="
  (cd "$dev" && find . -type f | sort)

  # --- CocoaPods form ---
  p="$work/pods/$name"
  mkdir -p "$p/device" "$p/sim"
  cp -R "$dev" "$p/device/"
  cp -R "$sim_arm64" "$p/sim/"
  lipo -create "$sim_arm64/$name" "$sim_x86/$name" -output "$p/sim/$name.framework/$name"
  rm -f "$p/device/$name.framework/Info.plist" "$p/sim/$name.framework/Info.plist"
  rm -rf "$out/pods/$name.xcframework"
  if ! xcodebuild -create-xcframework \
        -framework "$p/device/$name.framework" \
        -framework "$p/sim/$name.framework" \
        -output "$out/pods/$name.xcframework"; then
    echo "xcodebuild refused bundle-less frameworks; assembling $name.xcframework by hand"
    rm -rf "$out/pods/$name.xcframework"
    mkdir -p "$out/pods/$name.xcframework/ios-arm64" \
             "$out/pods/$name.xcframework/ios-arm64_x86_64-simulator"
    cp -R "$p/device/$name.framework" "$out/pods/$name.xcframework/ios-arm64/"
    cp -R "$p/sim/$name.framework" "$out/pods/$name.xcframework/ios-arm64_x86_64-simulator/"
    write_pods_xcframework_plist "$name" "$out/pods/$name.xcframework/Info.plist"
  fi
  # Same layout as libs-v0.1.9: no per-bundle Info.plist in the CocoaPods form.
  find "$out/pods/$name.xcframework" -path "*/$name.framework/Info.plist" -delete

  # --- SwiftPM form ---
  s="$work/spm/$name"
  mkdir -p "$s/device" "$s/sim"
  cp -R "$dev" "$s/device/"
  cp -R "$sim_arm64" "$s/sim/"
  # Thin, as today's SPM zips: lipo -thin arm64 of the universal simulator binary.
  lipo -thin arm64 "$p/sim/$name.framework/$name" -output "$s/sim/$name.framework/$name"
  write_bundle_plist "$name" iPhoneOS "$s/device/$name.framework/Info.plist"
  write_bundle_plist "$name" iPhoneSimulator "$s/sim/$name.framework/Info.plist"
  rm -rf "$out/spm/$name.xcframework"
  xcodebuild -create-xcframework \
    -framework "$s/device/$name.framework" \
    -framework "$s/sim/$name.framework" \
    -output "$out/spm/$name.xcframework"
  plutil -p "$out/spm/$name.xcframework/Info.plist"
  rm -f "$out/$name-spm.xcframework.zip"
  (cd "$out/spm" && zip -qry "$out/$name-spm.xcframework.zip" "$name.xcframework")
  sha_file "$out/$name-spm.xcframework.zip"
done

if [ -n "$coreml" ]; then
  rm -rf "$out/pods/TensorFlowLiteCCoreML.xcframework"
  cp -R "$coreml" "$out/pods/"
fi

rm -f "$out/ios-frameworks.zip"
(cd "$out/pods" && zip -qry "$out/ios-frameworks.zip" ./*.xcframework)
sha_file "$out/ios-frameworks.zip"
unzip -l "$out/ios-frameworks.zip" | grep -v '/Headers/' | sed -n '1,80p'
