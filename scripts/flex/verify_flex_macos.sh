#!/bin/bash
# Verifies a freshly built libtensorflowlite_flex-mac.dylib against the
# previous release. Used by .github/workflows/build-flex-macos.yml.
#
# Usage: verify_flex_macos.sh <new dylib> <report dir>
# Env:   REF_DYLIB_URL REF_DYLIB_SHA256  previous dylib (flex-v1.0.0)
#        EXPECT_VERSION                  e.g. 2.21.0
#        EXPECT_MINOS                    e.g. 12.0
#
# Hard failures: not arm64, minimum OS not EXPECT_MINOS, exports other than
# the two plugin entry points, an undefined symbol that is not a system one,
# a link to a non-system library, missing version string, bad signature.
set -euo pipefail

NEW="$1"
REPORT="$(mkdir -p "$2" && cd "$2" && pwd)"
: "${REF_DYLIB_URL:?}" "${REF_DYLIB_SHA256:?}" "${EXPECT_VERSION:?}" "${EXPECT_MINOS:?}"
W="$(mktemp -d)"
OUT="$REPORT/verify-macos.txt"
: > "$OUT"
log() { echo "$@" | tee -a "$OUT"; }
err() { log "::error::$*"; fail=1; }
fail=0

curl -sSfL --retry 3 "$REF_DYLIB_URL" -o "$W/ref.dylib"
echo "$REF_DYLIB_SHA256  $W/ref.dylib" | shasum -a 256 -c -
REF="$W/ref.dylib"

for which in new ref; do
  if [ "$which" = new ]; then f="$NEW"; else f="$REF"; fi
  log "== $which: $(stat -f %z "$f") bytes, sha256 $(shasum -a 256 "$f" | cut -d' ' -f1)"
  log "   file: $(file -b "$f")"
  log "   archs: $(lipo -archs "$f")"
  log "   install name: $(otool -D "$f" | tail -1)"
  log "   build version: $(otool -l "$f" | awk '/cmd LC_BUILD_VERSION/{b=1} b&&/platform|minos|sdk/{printf "%s %s ", $1, $2} b&&/sdk/{b=0}')"
  log "   linked libraries:"
  otool -L "$f" | tail -n +2 | sed 's/^/     /' | tee -a "$OUT" > /dev/null
  nm -gU -j "$f" | sort -u > "$REPORT/exports-$which.txt"
  nm -u -j "$f" | sort -u > "$REPORT/undefined-$which.txt"
  log "   exports: $(wc -l < "$REPORT/exports-$which.txt" | tr -d ' '): $(tr '\n' ' ' < "$REPORT/exports-$which.txt")"
  log "   undefined: $(wc -l < "$REPORT/undefined-$which.txt" | tr -d ' ')"
  log "   symbol table entries: $(nm "$f" | wc -l | tr -d ' ')"
  log "   version strings: $(LC_ALL=C strings -a "$f" | LC_ALL=C grep -E '^2\.[0-9]+\.[0-9]+([-+][A-Za-z0-9.+-]*)?$' | sort -u | tr '\n' ' ' || true)"
  log "   codesign: $(codesign -dv "$f" 2>&1 | grep -E 'Signature|flags' | tr '\n' ' ' || true)"
done

log ""
log "== checks on the new dylib"
[ "$(lipo -archs "$NEW")" = arm64 ] || err "archs are $(lipo -archs "$NEW"), expected arm64"
minos="$(otool -l "$NEW" | awk '/cmd LC_BUILD_VERSION/{b=1} b&&$1=="minos"{print $2; exit}')"
[ "$minos" = "$EXPECT_MINOS" ] || err "minos is $minos, expected $EXPECT_MINOS"
printf '_tflite_plugin_create_delegate\n_tflite_plugin_destroy_delegate\n' > "$W/want-exports"
if diff "$W/want-exports" "$REPORT/exports-new.txt" > /dev/null; then
  log "exports: exactly the two plugin entry points"
else
  err "exports differ from the two plugin entry points (see exports-new.txt)"
fi

comm -13 "$REPORT/undefined-ref.txt" "$REPORT/undefined-new.txt" > "$REPORT/undefined-added.txt"
comm -23 "$REPORT/undefined-ref.txt" "$REPORT/undefined-new.txt" > "$REPORT/undefined-removed.txt"
log "undefined vs 2.20: added $(wc -l < "$REPORT/undefined-added.txt" | tr -d ' '), removed $(wc -l < "$REPORT/undefined-removed.txt" | tr -d ' ')"
log "  added: $(tr '\n' ' ' < "$REPORT/undefined-added.txt")"
log "  removed: $(tr '\n' ' ' < "$REPORT/undefined-removed.txt")"
# The TF .bazelrc macos config links with -undefined dynamic_lookup, so a
# missing TF/TFLite object would only show up here, not as a link error.
if LC_ALL=C grep -E 'TfLite|tflite|6tflite|10tensorflow|xnn|XNN|pthreadpool|absl|protobuf|_TF_|_TFE_' \
    "$REPORT/undefined-new.txt" > "$REPORT/undefined-non-system.txt"; then
  err "undefined TF/TFLite-like symbols: $(tr '\n' ' ' < "$REPORT/undefined-non-system.txt" | cut -c1-2000)"
else
  log "undefined: no TF/TFLite-like symbols"
fi
self_id="$(otool -D "$NEW" | tail -1)"
nonsys="$(otool -L "$NEW" | tail -n +2 | awk '{print $1}' | grep -vxF "$self_id" \
  | grep -vE '^(/usr/lib/|/System/Library/)' || true)"
[ -z "$nonsys" ] || err "links non-system libraries: $nonsys"
if LC_ALL=C strings -a "$NEW" | LC_ALL=C grep -E "^${EXPECT_VERSION//./\\.}([-+]|$)" > /dev/null; then
  log "version string $EXPECT_VERSION: present"
else
  err "version string $EXPECT_VERSION not found"
fi
codesign -v "$NEW" && log "codesign -v: ok" || err "codesign -v failed"

rm -rf "$W"
exit "$fail"
