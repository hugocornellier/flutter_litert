#!/bin/bash
# Makes TensorFlowLiteFlex-spm.xcframework.zip, the SwiftPM binary target that
# flutter_litert_flex's ios/flutter_litert_flex/Package.swift consumes, from a
# freshly built TensorFlowLiteFlex-ios.xcframework.zip.
#
# 1. Runs flutter_litert_flex's own scripts/preprocess_flex_xcframework.sh,
#    pinned by commit and sha256, with its download line replaced by a copy of
#    the local zip and its dedup pattern widened if needed (see below). It dedups
#    Bazel's duplicate .o members, drops members that
#    need WebP/ICU/protobuf-compiler symbols, and drops the members that define
#    the XNNPack delegate (TensorFlowLiteC provides those).
# 2. Repackages the result as a static-library xcframework
#    (libTensorFlowLiteFlex.a per slice), the form published as
#    flex-v1.1.1/TensorFlowLiteFlex-spm.xcframework.zip. SwiftPM links a
#    -library xcframework with -l, so -all_load reaches it; a .framework one is
#    only embedded (see the comment in the flex Package.swift).
#
# Usage: make_flex_ios_spm.sh <TensorFlowLiteFlex-ios.xcframework.zip> <out dir>
# Env:   FLEX_REPO_COMMIT, PREPROCESS_SHA256 (pins for the preprocess script)
# Output: <out dir>/TensorFlowLiteFlex-spm.xcframework.zip (+ .sha256)
set -euo pipefail

RAW_ZIP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
: "${FLEX_REPO_COMMIT:?}" "${PREPROCESS_SHA256:?}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

script_url="https://raw.githubusercontent.com/hugocornellier/flutter_litert_flex/${FLEX_REPO_COMMIT}/scripts/preprocess_flex_xcframework.sh"
curl -sSfL --retry 3 "$script_url" -o "$WORK/preprocess.sh"
echo "${PREPROCESS_SHA256}  $WORK/preprocess.sh" | shasum -a 256 -c -

# What changes in the fetched script:
# - the download becomes a copy of the local zip;
# - the dedup's hash-suffix pattern also accepts 16 hex digits. Bazel 7.7.0
#   (TF 2.21.0) names its second copy of a same-named object
#   node_def_util_3e6b16f69e25a5d1.o; the dedup was written for 32 digits.
#   With only {32} the script groups none of the 2.21 copies, keeps both, and
#   an -all_load link fails with about 2600 duplicate symbols (run
#   42656958304).
#
# The pin may be on either side of flutter_litert_flex commit ea29249, which
# made the same pattern change in the flex repo (script and podspec):
# - before it (af65014, the pin the flex-v2.0.0 files were built with) the
#   pattern is the {32} one and is widened here;
# - from it on the pattern is already the widened one and is left alone. From
#   563e121 on the script also checks its download against DOWNLOAD_SHA256
#   (the published zip), so it is run with DOWNLOAD_SHA256 set to the local
#   zip's own hash. An older script does not read that variable.
download_line='curl -sL "$DOWNLOAD_URL" -o "$WORK_DIR/raw.zip"'
test "$(grep -cF "$download_line" "$WORK/preprocess.sh")" = 1
python3 -I - "$WORK/preprocess.sh" "$WORK/preprocess_local.sh" "$download_line" "$RAW_ZIP" <<'PY'
import sys, shlex
src, dst, old, raw = sys.argv[1:5]
text = open(src).read()
new = 'cp ' + shlex.quote(raw) + ' "$WORK_DIR/raw.zip"'
assert text.count(old) == 1
text = text.replace(old, new)
old_re = r"re.sub(r'_[a-f0-9]{32}\.o$', '.o', raw)"
new_re = r"re.sub(r'_(?:[a-f0-9]{16}|[a-f0-9]{32})\.o$', '.o', raw)"
if text.count(old_re) == 1:
    text = text.replace(old_re, new_re)
assert text.count(old_re) == 0 and text.count(new_re) == 1
open(dst, 'w').write(text)
PY
diff "$WORK/preprocess.sh" "$WORK/preprocess_local.sh" || true

RAW_SHA256="$(shasum -a 256 "$RAW_ZIP" | awk '{print $1}')"
mkdir -p "$WORK/run"
(cd "$WORK/run" && DOWNLOAD_SHA256="$RAW_SHA256" bash "$WORK/preprocess_local.sh")
FW_ZIP="$WORK/run/TensorFlowLiteFlex-spm.xcframework.zip"
test -s "$FW_ZIP"

# Repackage as a static-library xcframework.
mkdir -p "$WORK/fw" "$WORK/lib/ios-arm64" "$WORK/lib/ios-arm64-simulator"
unzip -q "$FW_ZIP" -d "$WORK/fw"
for id in ios-arm64 ios-arm64-simulator; do
  bin="$WORK/fw/TensorFlowLiteFlex.xcframework/$id/TensorFlowLiteFlex.framework/TensorFlowLiteFlex"
  test -s "$bin"
  mv "$bin" "$WORK/lib/$id/libTensorFlowLiteFlex.a"
done
rm -rf "$WORK/fw"

rm -rf "$OUT_DIR/TensorFlowLiteFlex.xcframework"
xcodebuild -create-xcframework \
  -library "$WORK/lib/ios-arm64/libTensorFlowLiteFlex.a" \
  -library "$WORK/lib/ios-arm64-simulator/libTensorFlowLiteFlex.a" \
  -output "$OUT_DIR/TensorFlowLiteFlex.xcframework"
rm -rf "$WORK/lib"

(cd "$OUT_DIR" && rm -f TensorFlowLiteFlex-spm.xcframework.zip \
  && zip -qr TensorFlowLiteFlex-spm.xcframework.zip TensorFlowLiteFlex.xcframework \
  && shasum -a 256 TensorFlowLiteFlex-spm.xcframework.zip > TensorFlowLiteFlex-spm.xcframework.zip.sha256)
cat "$OUT_DIR/TensorFlowLiteFlex-spm.xcframework.zip.sha256"
cat "$OUT_DIR/TensorFlowLiteFlex.xcframework/Info.plist"
