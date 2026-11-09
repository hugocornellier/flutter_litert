#!/bin/bash
# Makes flutter_litert_flex-macos-spm.xcframework.zip, the SwiftPM binary
# target of flutter_litert_flex on macOS, from libtensorflowlite_flex-mac.dylib.
#
# Reproduces the layout of flex-v1.1.0/flutter_litert_flex-macos-spm.xcframework.zip
# (made by hand for flutter_litert_flex commit 3fdbe48, "Add SPM support for iOS
# and macOS"; no script existed):
#
#   flutter_litert_flex.xcframework/
#     Info.plist                       (xcodebuild -create-xcframework -framework)
#     macos-arm64/flutter_litert_flex.framework/
#       flutter_litert_flex -> Versions/Current/flutter_litert_flex
#       Resources -> Versions/Current/Resources
#       Versions/Current -> A
#       Versions/A/flutter_litert_flex           empty arm64 stub dylib
#       Versions/A/Resources/Info.plist
#       Versions/A/Resources/libtensorflowlite_flex-mac.dylib
#
# The stub only gives SwiftPM a framework binary to embed. The real library
# is a resource, which Dart opens by path:
#   Frameworks/flutter_litert_flex.framework/Versions/A/Resources/libtensorflowlite_flex-mac.dylib
# (lib/src/delegates/flex_delegate_native.dart in flutter_litert_flex).
#
# Usage: make_flex_macos_spm.sh <libtensorflowlite_flex-mac.dylib> <out dir>
# Env:   STUB_MIN_MACOS (default 11.0, as in the published stub)
#        BUNDLE_SHORT_VERSION (default 0.0.5, as in the published Info.plist)
# Output: <out dir>/flutter_litert_flex-macos-spm.xcframework.zip (+ .sha256)
set -euo pipefail

DYLIB="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
STUB_MIN_MACOS="${STUB_MIN_MACOS:-11.0}"
BUNDLE_SHORT_VERSION="${BUNDLE_SHORT_VERSION:-0.0.5}"

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
FW="$W/flutter_litert_flex.framework"
mkdir -p "$FW/Versions/A/Resources"

# Empty stub binary.
: > "$W/stub.c"
xcrun clang -dynamiclib -arch arm64 -mmacosx-version-min="$STUB_MIN_MACOS" \
  -install_name @rpath/flutter_litert_flex.framework/Versions/A/flutter_litert_flex \
  "$W/stub.c" -o "$FW/Versions/A/flutter_litert_flex"

cp "$DYLIB" "$FW/Versions/A/Resources/libtensorflowlite_flex-mac.dylib"
# Same keys and values as the published Info.plist.
cat > "$FW/Versions/A/Resources/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleIdentifier</key>
	<string>com.hugocornellier.flutter-litert-flex</string>
	<key>CFBundleName</key>
	<string>flutter_litert_flex</string>
	<key>CFBundlePackageType</key>
	<string>FMWK</string>
	<key>CFBundleShortVersionString</key>
	<string>${BUNDLE_SHORT_VERSION}</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>MinimumOSVersion</key>
	<string>10.14</string>
</dict>
</plist>
PLIST
plutil -lint "$FW/Versions/A/Resources/Info.plist"

ln -s A "$FW/Versions/Current"
ln -s Versions/Current/flutter_litert_flex "$FW/flutter_litert_flex"
ln -s Versions/Current/Resources "$FW/Resources"

# Sign the stub so the framework bundle has a valid signature to embed; the
# dylib inside Resources keeps its own ad-hoc signature.
codesign -f -s - "$FW/Versions/A/flutter_litert_flex"

rm -rf "$OUT_DIR/flutter_litert_flex.xcframework"
xcodebuild -create-xcframework -framework "$FW" \
  -output "$OUT_DIR/flutter_litert_flex.xcframework"

(cd "$OUT_DIR" && rm -f flutter_litert_flex-macos-spm.xcframework.zip \
  && zip -qry flutter_litert_flex-macos-spm.xcframework.zip flutter_litert_flex.xcframework \
  && shasum -a 256 flutter_litert_flex-macos-spm.xcframework.zip \
       > flutter_litert_flex-macos-spm.xcframework.zip.sha256)
cat "$OUT_DIR/flutter_litert_flex-macos-spm.xcframework.zip.sha256"
zipinfo "$OUT_DIR/flutter_litert_flex-macos-spm.xcframework.zip"
