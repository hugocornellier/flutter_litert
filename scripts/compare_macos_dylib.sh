#!/bin/sh
# Compare a newly built macOS dylib against the committed one it replaces.
#
# Usage: compare_macos_dylib.sh <new.dylib> <old.dylib> <out-dir>
#
# For every architecture of the new file, prints the install name, minimum
# macOS, SDK, linked libraries and code signature, and diffs the exported
# (defined, external) symbol list against the same architecture of the old
# file. Writes <out-dir>/<arch>.exports.{new,old,added,removed}.txt and a
# summary to stdout. Fails if an architecture of the old file is missing from
# the new one, or if the new file's minimum macOS is higher than the old one's.
set -eu

new="$1"
old="$2"
out="$3"
mkdir -p "$out"

minos() {
  # LC_BUILD_VERSION (minos) or the older LC_VERSION_MIN_MACOSX (version).
  otool -arch "$1" -l "$2" | awk '
    /LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX/ {f=1}
    f && ($1=="minos" || $1=="version") {print $2; exit}'
}

ver_le() {
  # true when $1 <= $2 as dotted versions
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)" = "$1" ]
}

echo "new: $new"
shasum -a 256 "$new"
echo "old: $old"
shasum -a 256 "$old"
echo "new archs: $(lipo -archs "$new")"
echo "old archs: $(lipo -archs "$old")"

status=0
for a in $(lipo -archs "$old"); do
  if ! lipo -archs "$new" | tr ' ' '\n' | grep -qx "$a"; then
    echo "FAIL: arch $a is in the old file but not the new one"
    status=1
  fi
done

for a in $(lipo -archs "$new"); do
  echo
  echo "=== $a ==="
  echo "install name: $(otool -arch "$a" -D "$new" | tail -n +2)"
  nm_new=$(minos "$a" "$new")
  echo "minos new: $nm_new"
  if lipo -archs "$old" | tr ' ' '\n' | grep -qx "$a"; then
    nm_old=$(minos "$a" "$old")
    echo "minos old: $nm_old"
    echo "install name old: $(otool -arch "$a" -D "$old" | tail -n +2)"
    if ! ver_le "$nm_new" "$nm_old"; then
      echo "FAIL: minimum macOS rose from $nm_old to $nm_new"
      status=1
    fi
  fi
  echo "linked libraries:"
  otool -arch "$a" -L "$new" | tail -n +2
  echo "version-like strings:"
  strings -arch "$a" "$new" | grep -E '^2\.[0-9]+\.[0-9]+' | head -5 || true
  nm -arch "$a" -gU "$new" | awk '{print $NF}' | sort -u > "$out/$a.exports.new.txt"
  echo "exports new: $(wc -l < "$out/$a.exports.new.txt" | tr -d ' ')"
  if lipo -archs "$old" | tr ' ' '\n' | grep -qx "$a"; then
    nm -arch "$a" -gU "$old" | awk '{print $NF}' | sort -u > "$out/$a.exports.old.txt"
    comm -13 "$out/$a.exports.old.txt" "$out/$a.exports.new.txt" > "$out/$a.exports.added.txt"
    comm -23 "$out/$a.exports.old.txt" "$out/$a.exports.new.txt" > "$out/$a.exports.removed.txt"
    echo "exports old: $(wc -l < "$out/$a.exports.old.txt" | tr -d ' ')"
    echo "added ($(wc -l < "$out/$a.exports.added.txt" | tr -d ' ')):"
    head -50 "$out/$a.exports.added.txt"
    echo "removed ($(wc -l < "$out/$a.exports.removed.txt" | tr -d ' ')):"
    head -50 "$out/$a.exports.removed.txt"
  fi
done

echo
codesign -dv "$new" 2>&1 || true
exit $status
