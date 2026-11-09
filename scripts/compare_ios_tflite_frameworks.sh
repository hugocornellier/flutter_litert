#!/usr/bin/env bash
# Compares a newly built iOS xcframework against the published one it replaces.
#
# Usage:
#   scripts/compare_ios_tflite_frameworks.sh <new.xcframework> <ref.xcframework> <expected-version> [report-dir]
#
# For every slice directory of the reference (ios-arm64, ios-arm64-simulator,
# ios-arm64_x86_64-simulator) and every architecture in it, checks that:
#   - the new xcframework has the same slice and the same architectures;
#   - each architecture has the same platform and minimum OS (the SDK version
#     is printed but may differ: it follows the runner's Xcode);
#   - if the reference binary embeds a TF version string (2.x.y-...), the new
#     binary embeds one that starts with <expected-version>.
# Fails (exit 1) on any of those. Also prints, without failing:
#   - the exported (global, defined) symbol diff per architecture, as added
#     and removed lists;
#   - a diff of Headers/, Modules/ and the per-bundle Info.plist.
#
# With a report-dir, the symbol lists and diffs are written there as files.
# Inspection only: lipo, otool, nm, strings. Nothing is executed.
set -euo pipefail

new="$1"
ref="$2"
expected="$3"
report="${4:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
[ -n "$report" ] && mkdir -p "$report"
fail=0

name="$(basename "$new" .xcframework)"

# Prints "platform=P minos=M" per architecture (one line per distinct value).
build_versions() {
  otool -arch "$2" -l "$1" | awk '
    /cmd LC_BUILD_VERSION/ {f=1; v=0; next}
    /cmd LC_VERSION_MIN_IPHONEOS/ {v=1; f=0; next}
    f && /platform/ {p=$2}
    f && /minos/ {m=$2; printf "platform=%s minos=%s\n", p, m; f=0}
    v && /version/ {m=$2; printf "LC_VERSION_MIN_IPHONEOS minos=%s\n", m; v=0}
  ' | sort -u
}

sdks() {
  otool -arch "$2" -l "$1" | awk '/^ *sdk / {print $2}' | sort -u | tr '\n' ' '
}

echo "##### $name: new=$new ref=$ref"
for slice_dir in "$ref"/*/; do
  slice="$(basename "$slice_dir")"
  ref_bin="$ref/$slice/$name.framework/$name"
  new_bin="$new/$slice/$name.framework/$name"
  [ -f "$ref_bin" ] || continue
  echo "=== slice $slice"
  if [ ! -f "$new_bin" ]; then
    echo "FAIL: new xcframework has no $slice/$name.framework/$name"
    fail=1
    continue
  fi
  ref_archs="$(lipo -archs "$ref_bin")"
  new_archs="$(lipo -archs "$new_bin")"
  echo "archs ref: $ref_archs | new: $new_archs"
  if [ "$(echo "$ref_archs" | tr ' ' '\n' | sort | tr '\n' ' ')" != \
       "$(echo "$new_archs" | tr ' ' '\n' | sort | tr '\n' ' ')" ]; then
    echo "FAIL: architectures differ"
    fail=1
  fi
  echo "size ref: $(wc -c < "$ref_bin" | tr -d ' ') | new: $(wc -c < "$new_bin" | tr -d ' ')"
  echo "sha256 new: $(shasum -a 256 "$new_bin" | awk '{print $1}')"

  for arch in $ref_archs; do
    echo "--- $slice [$arch]"
    rv="$(build_versions "$ref_bin" "$arch")"
    nv="$(build_versions "$new_bin" "$arch" 2>/dev/null || true)"
    echo "build version ref: $(echo "$rv" | tr '\n' ';') sdk $(sdks "$ref_bin" "$arch")"
    echo "build version new: $(echo "$nv" | tr '\n' ';') sdk $(sdks "$new_bin" "$arch" 2>/dev/null || true)"
    if [ "$rv" != "$nv" ]; then
      echo "FAIL: platform or minimum OS differs"
      fail=1
    fi
    nm -arch "$arch" -gU "$ref_bin" 2>/dev/null | awk 'NF>=3 {print $3}' | sort -u > "$work/ref.syms"
    nm -arch "$arch" -gU "$new_bin" 2>/dev/null | awk 'NF>=3 {print $3}' | sort -u > "$work/new.syms" || true
    added="$(comm -13 "$work/ref.syms" "$work/new.syms")"
    removed="$(comm -23 "$work/ref.syms" "$work/new.syms")"
    echo "global symbols ref: $(wc -l < "$work/ref.syms" | tr -d ' ') | new: $(wc -l < "$work/new.syms" | tr -d ' ')"
    echo "added ($(printf '%s' "$added" | grep -c . || true)):"
    printf '%s\n' "$added" | sed '/^$/d; s/^/    + /'
    echo "removed ($(printf '%s' "$removed" | grep -c . || true)):"
    printf '%s\n' "$removed" | sed '/^$/d; s/^/    - /'
    if [ -n "$report" ]; then
      cp "$work/new.syms" "$report/$name.$slice.$arch.new.syms"
      cp "$work/ref.syms" "$report/$name.$slice.$arch.ref.syms"
    fi
  done

  ref_ver="$(strings -a "$ref_bin" | grep -E '^2\.[0-9]+\.[0-9]+(-[A-Za-z0-9.+]+)?$' | sort -u | tr '\n' ' ' || true)"
  new_ver="$(strings -a "$new_bin" | grep -E '^2\.[0-9]+\.[0-9]+(-[A-Za-z0-9.+]+)?$' | sort -u | tr '\n' ' ' || true)"
  echo "version strings ref: [$ref_ver] new: [$new_ver]"
  if [ -n "$ref_ver" ]; then
    exp_re="^${expected//./\\.}(-|\$)"
    if ! printf '%s\n' $new_ver | grep -Eq "$exp_re"; then
      echo "FAIL: no version string starting with $expected"
      fail=1
    fi
    if printf '%s\n' $new_ver | grep -Ev "$exp_re" | grep -q 'selfbuilt'; then
      echo "FAIL: an older self-built version string is still embedded"
      fail=1
    fi
  fi

  echo "--- $slice framework files other than the binary (diff ref -> new)"
  (cd "$ref/$slice/$name.framework" && find . -type f ! -name "$name" | sort) > "$work/ref.files"
  (cd "$new/$slice/$name.framework" && find . -type f ! -name "$name" | sort) > "$work/new.files"
  diff "$work/ref.files" "$work/new.files" && echo "same file list"
  for d in Headers Modules; do
    if [ -d "$ref/$slice/$name.framework/$d" ] || [ -d "$new/$slice/$name.framework/$d" ]; then
      if diff -ru "$ref/$slice/$name.framework/$d" "$new/$slice/$name.framework/$d" > "$work/$d.diff"; then
        echo "$d: identical"
      else
        echo "$d: differs ($(wc -l < "$work/$d.diff" | tr -d ' ') diff lines)"
        sed -n '1,60p' "$work/$d.diff"
        [ -n "$report" ] && cp "$work/$d.diff" "$report/$name.$slice.$d.diff"
      fi
    fi
  done
  if [ -f "$ref/$slice/$name.framework/Info.plist" ] || [ -f "$new/$slice/$name.framework/Info.plist" ]; then
    diff <(plutil -p "$ref/$slice/$name.framework/Info.plist" 2>&1) \
         <(plutil -p "$new/$slice/$name.framework/Info.plist" 2>&1) \
      && echo "bundle Info.plist: identical" || true
  fi
done

for slice_dir in "$new"/*/; do
  slice="$(basename "$slice_dir")"
  [ -d "$ref/$slice" ] || { echo "FAIL: new xcframework has extra slice $slice"; fail=1; }
done

echo "--- xcframework Info.plist (diff ref -> new)"
diff <(plutil -p "$ref/Info.plist") <(plutil -p "$new/Info.plist") && echo "identical" || true

if [ "$fail" -ne 0 ]; then
  echo "RESULT $name: FAIL"
  exit 1
fi
echo "RESULT $name: OK"
