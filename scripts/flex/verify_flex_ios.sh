#!/bin/bash
# Verifies a freshly built iOS TensorFlowLiteFlex (raw framework xcframework
# and SPM static-library xcframework) against the previous release.
# Used by .github/workflows/build-flex-ios.yml.
#
# Usage: verify_flex_ios.sh <new TensorFlowLiteFlex-ios.xcframework.zip>
#                           <new TensorFlowLiteFlex-spm.xcframework.zip>
#                           <report dir>
# Env (pins, all required):
#   REF_RAW_URL REF_RAW_SHA256     previous raw xcframework zip (flex-v1.0.0)
#   REF_SPM_URL REF_SPM_SHA256     previous SPM xcframework zip (flex-v1.1.1)
#   TFLC_SPM_URL TFLC_SPM_SHA256   TensorFlowLiteC SPM xcframework zip the
#                                  flex archive is linked with under SwiftPM
#   EXPECT_VERSION                 e.g. 2.21.0
#
# Hard failures: wrong archs, missing plugin entry points, duplicate symbols
# in the SwiftPM-form link. Everything else (minimum OS, version string,
# symbol diffs, undefined symbols in the link) is reported for a human to
# judge.
set -euo pipefail

NEW_RAW_ZIP="$1"
NEW_SPM_ZIP="$2"
REPORT="$(mkdir -p "$3" && cd "$3" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${REF_RAW_URL:?}" "${REF_RAW_SHA256:?}" "${REF_SPM_URL:?}" "${REF_SPM_SHA256:?}"
: "${TFLC_SPM_URL:?}" "${TFLC_SPM_SHA256:?}" "${EXPECT_VERSION:?}"

W="$(mktemp -d)"
fail=0

fetch() { # url sha out
  curl -sSfL --retry 3 "$1" -o "$3"
  echo "$2  $3" | shasum -a 256 -c -
}

echo "== fetching references"
fetch "$REF_RAW_URL" "$REF_RAW_SHA256" "$W/ref-raw.zip"
fetch "$REF_SPM_URL" "$REF_SPM_SHA256" "$W/ref-spm.zip"
fetch "$TFLC_SPM_URL" "$TFLC_SPM_SHA256" "$W/tflc-spm.zip"
mkdir -p "$W/new-raw" "$W/new-spm" "$W/ref-raw" "$W/ref-spm" "$W/tflc"
unzip -q "$NEW_RAW_ZIP" -d "$W/new-raw"
unzip -q "$NEW_SPM_ZIP" -d "$W/new-spm"
# Previous releases: only the device slice is needed for symbol work.
unzip -q "$W/ref-raw.zip" 'TensorFlowLiteFlex.xcframework/ios-arm64/*' -d "$W/ref-raw"
unzip -q "$W/ref-spm.zip" 'TensorFlowLiteFlex.xcframework/ios-arm64/*' -d "$W/ref-spm"
unzip -q "$W/tflc-spm.zip" -d "$W/tflc"
rm -f "$W/ref-raw.zip" "$W/ref-spm.zip" "$W/tflc-spm.zip"
df -h / | tail -1

NR="$W/new-raw/TensorFlowLiteFlex.xcframework"
NS="$W/new-spm/TensorFlowLiteFlex.xcframework"
RR="$W/ref-raw/TensorFlowLiteFlex.xcframework/ios-arm64/TensorFlowLiteFlex.framework/TensorFlowLiteFlex"
RS="$W/ref-spm/TensorFlowLiteFlex.xcframework/ios-arm64/libTensorFlowLiteFlex.a"
TC="$W/tflc/TensorFlowLiteC.xcframework/ios-arm64/TensorFlowLiteC.framework"

declare -a BINS=(
  "raw ios-arm64|$NR/ios-arm64/TensorFlowLiteFlex.framework/TensorFlowLiteFlex"
  "raw ios-arm64-simulator|$NR/ios-arm64-simulator/TensorFlowLiteFlex.framework/TensorFlowLiteFlex"
  "spm ios-arm64|$NS/ios-arm64/libTensorFlowLiteFlex.a"
  "spm ios-arm64-simulator|$NS/ios-arm64-simulator/libTensorFlowLiteFlex.a"
)

build_info() { # binary -> "platform minos sdk" histogram over all members
  otool -l "$1" | awk '
    /cmd LC_BUILD_VERSION/ {bv=1; next}
    /cmd LC_VERSION_MIN_IPHONEOS/ {vm=1; p="IOS(min_cmd)"; next}
    bv && $1=="platform" {p=$2}
    bv && $1=="minos" {m=$2}
    bv && $1=="sdk" {print p, "minos", m, "sdk", $2; bv=0}
    vm && $1=="version" {m=$2}
    vm && $1=="sdk" {print p, "minos", m, "sdk", $2; vm=0}
  ' | sort | uniq -c
}

{
  echo "== xcframework layout"
  for x in "$NR" "$NS"; do
    echo "-- $x"
    (cd "$x" && find . -maxdepth 3 | sort)
    plutil -p "$x/Info.plist"
  done

  echo
  echo "== archs, sizes, member build versions"
  for e in "${BINS[@]}"; do
    label="${e%%|*}"; bin="${e#*|}"
    echo "-- $label: $(stat -f %z "$bin") bytes, sha256 $(shasum -a 256 "$bin" | cut -d' ' -f1)"
    archs="$(lipo -archs "$bin")"
    echo "   archs: $archs"
    [ "$archs" = "arm64" ] || { echo "::error::$label archs are '$archs', expected arm64"; fail=1; }
    build_info "$bin" | sed 's/^/   /'
  done
  echo "-- reference raw ios-arm64 (flex-v1.0.0)"
  echo "   archs: $(lipo -archs "$RR")"
  build_info "$RR" | sed 's/^/   /'

  echo
  echo "== hash-suffixed object members (Bazel's second copy of a same-named .o)"
  # The podspec and preprocess dedup of flutter_litert_flex before 2.1.0
  # match only a 32-digit suffix.
  for e in "${BINS[@]}" "ref raw ios-arm64|$RR" "ref spm ios-arm64|$RS"; do
    label="${e%%|*}"; bin="${e#*|}"
    ar -t "$bin" > "$W/members.txt"
    echo "-- $label: members $(grep -c '\.o$' "$W/members.txt" || true)," \
      "_<16 hex>.o $(grep -cE '_[0-9a-f]{16}\.o$' "$W/members.txt" || true)," \
      "_<32 hex>.o $(grep -cE '_[0-9a-f]{32}\.o$' "$W/members.txt" || true)"
  done

  echo
  echo "== version strings (TF / TFLite)"
  for e in "${BINS[@]:0:1}" "ref raw ios-arm64|$RR"; do
    label="${e%%|*}"; bin="${e#*|}"
    v="$(LC_ALL=C strings -a "$bin" | LC_ALL=C grep -E '^2\.[0-9]+\.[0-9]+([-+.][A-Za-z0-9.+-]*)?$' | sort | uniq -c | sort -rn | head -5 || true)"
    echo "-- $label:"; echo "$v" | sed 's/^/   /'
  done
  if ! LC_ALL=C strings -a "$NR/ios-arm64/TensorFlowLiteFlex.framework/TensorFlowLiteFlex" \
      | LC_ALL=C grep -E "^${EXPECT_VERSION//./\\.}([-+]|$)" > /dev/null; then
    # Reported, not fatal: a static archive only carries the string when some
    # linked object references TF_VERSION_STRING. Compare with the reference.
    echo "::warning::version string $EXPECT_VERSION not found in the new raw device slice"
  fi

  echo
  echo "== plugin entry points in every slice"
  for e in "${BINS[@]}"; do
    label="${e%%|*}"; bin="${e#*|}"
    syms="$(nm -gUj "$bin" 2>/dev/null | LC_ALL=C grep -E '^_tflite_plugin_(create|destroy)_delegate$' | sort -u | tr '\n' ' ' || true)"
    echo "-- $label: $syms"
    case "$syms" in
      *_tflite_plugin_create_delegate*_tflite_plugin_destroy_delegate*) ;;
      *) echo "::error::$label lacks a plugin entry point"; fail=1 ;;
    esac
  done
} 2>&1 | tee "$REPORT/verify-ios.txt"
# tee hides the block's exit status; the error lines carry the result.
if grep -q '^::error::' "$REPORT/verify-ios.txt"; then fail=1; fi

echo
echo "== symbol analysis (device slices)" | tee -a "$REPORT/verify-ios.txt"
python3 -I "$HERE/flex_symbols.py" \
  --new-raw "$NR/ios-arm64/TensorFlowLiteFlex.framework/TensorFlowLiteFlex" \
  --new-spm "$NS/ios-arm64/libTensorFlowLiteFlex.a" \
  --ref-raw "$RR" --ref-spm "$RS" \
  --tflc "$TC/TensorFlowLiteC" \
  --out "$REPORT/symbols" 2>&1 | tee -a "$REPORT/verify-ios.txt" || fail=1
grep -q '^::error::' "$REPORT/verify-ios.txt" && fail=1

echo
echo "== SwiftPM-form link test" | tee -a "$REPORT/verify-ios.txt"
# Mirrors an app link under SwiftPM: the flex static library with -all_load
# (flutter_litert_flex Package.swift) plus the TensorFlowLiteC framework and
# the frameworks flutter_litert's Package.swift links. Links against the
# given TensorFlowLiteC; with the 2.20 one, missing 2.21 symbols are expected.
cat > "$W/main.c" <<'C'
#include <stddef.h>
typedef struct TfLiteDelegate TfLiteDelegate;
TfLiteDelegate* tflite_plugin_create_delegate(const char* const*, const char* const*,
                                              size_t, void (*)(const char*));
void tflite_plugin_destroy_delegate(TfLiteDelegate*);
const char* TfLiteVersion(void);
int main(void) {
  TfLiteDelegate* d = tflite_plugin_create_delegate(NULL, NULL, 0, NULL);
  tflite_plugin_destroy_delegate(d);
  return TfLiteVersion() ? 0 : 1;
}
C
# Duplicate symbols are a hard failure: they come from the flex archive
# itself (or from it and TensorFlowLiteC) whatever TensorFlowLiteC is given.
# Undefined symbols stay report-only. The 2.20 SPM archive (device slice) is
# linked the same way as a control.
mkdir -p "$W/ref-spm-lib/ios-arm64"
ln -s "$RS" "$W/ref-spm-lib/ios-arm64/libTensorFlowLiteFlex.a"
for e in "new|ios-arm64|$NS" "new|ios-arm64-simulator|$NS" "ref|ios-arm64|$W/ref-spm-lib"; do
  which="${e%%|*}"; rest="${e#*|}"; slice="${rest%%|*}"; libdir="${rest#*|}"
  if [ "$slice" = ios-arm64 ]; then sdk=iphoneos; tgt=arm64-apple-ios15.0
  else sdk=iphonesimulator; tgt=arm64-apple-ios15.0-simulator; fi
  log="$REPORT/link-spm-$which-$slice.txt"
  set +e
  xcrun --sdk "$sdk" clang -target "$tgt" "$W/main.c" -o "$W/app-$which-$slice" \
    -Wl,-all_load -L"$libdir/$slice" -lTensorFlowLiteFlex \
    -F"$W/tflc/TensorFlowLiteC.xcframework/$slice" -framework TensorFlowLiteC \
    -lc++ -framework Foundation -framework CoreFoundation -framework Security \
    -framework CoreML -framework Accelerate -framework Metal \
    > "$log" 2>&1
  rc=$?
  set -e
  undef="$(grep -c '^  "_' "$log" || true)"
  dup="$(grep -c 'duplicate symbol' "$log" || true)"
  # Filtered to a file first: grep into head under pipefail fails on SIGPIPE.
  grep -E '^  "_|duplicate symbol' "$log" > "$W/link-lines.txt" || true
  {
    echo "-- $which $slice: clang exit $rc"
    echo "   undefined: $undef duplicate: $dup"
    head -40 "$W/link-lines.txt" | cut -c1-240 | sed 's/^/   /'
    tail -3 "$log" | cut -c1-240 | sed 's/^/   /'
    if [ "$which" = new ] && [ "$dup" != 0 ]; then
      echo "::error::$slice SwiftPM-form link has $dup duplicate symbols"
    fi
  } | tee -a "$REPORT/verify-ios.txt"
  if [ "$which" = new ] && [ "$dup" != 0 ]; then fail=1; fi
done

rm -rf "$W"
exit "$fail"
