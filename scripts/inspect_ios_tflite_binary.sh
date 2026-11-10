#!/usr/bin/env bash
# Prints the facts used to compare an iOS TensorFlow Lite framework binary
# against the one it replaces: architectures, per-architecture platform,
# minimum OS and SDK (LC_BUILD_VERSION), the embedded TF version string, and
# the count of exported (global, defined) symbols.
#
# Usage: scripts/inspect_ios_tflite_binary.sh <framework binary> [symbols-out-prefix]
#
# With a symbols-out-prefix, writes <prefix>.<arch>.syms with the sorted list of
# global defined symbols for each architecture (input for a diff against the
# previous release).
#
# It only inspects (lipo, otool, nm, strings); it never runs the binary.
set -euo pipefail

bin="$1"
prefix="${2:-}"

echo "file: $bin"
echo "size: $(wc -c < "$bin" | tr -d ' ')"
echo "sha256: $(shasum -a 256 "$bin" | awk '{print $1}')"
archs=$(lipo -archs "$bin")
echo "archs: $archs"
for arch in $archs; do
  # LC_BUILD_VERSION: platform 2 = iOS, 7 = iOS simulator. Objects built for
  # a minimum below the platform's LC_BUILD_VERSION floor (x86_64 simulator at
  # iOS 12) carry LC_VERSION_MIN_IPHONEOS instead.
  bv=$(otool -arch "$arch" -l "$bin" | awk '
    /cmd LC_BUILD_VERSION/ {f=1; v=0; next}
    /cmd LC_VERSION_MIN_IPHONEOS/ {v=1; f=0; next}
    f && /platform/ {p=$2}
    f && /minos/ {m=$2}
    f && /sdk/ {s=$2; printf "platform=%s minos=%s sdk=%s\n", p, m, s; f=0}
    v && /version/ {m=$2}
    v && /sdk/ {s=$2; printf "LC_VERSION_MIN_IPHONEOS minos=%s sdk=%s\n", m, s; v=0}
  ' | sort | uniq -c | sort -rn)
  echo "[$arch] build version (count platform minos sdk):"
  echo "$bv" | sed 's/^/    /'
  ngl=$(nm -arch "$arch" -gU "$bin" 2>/dev/null | awk 'NF>=3 {print $3}' | sort -u | wc -l | tr -d ' ')
  echo "[$arch] global defined symbols: $ngl"
  if [ -n "$prefix" ]; then
    nm -arch "$arch" -gU "$bin" 2>/dev/null | awk 'NF>=3 {print $3}' | sort -u > "$prefix.$arch.syms"
  fi
done
echo "version strings:"
strings -a "$bin" | grep -E '^2\.[0-9]+\.[0-9]+(-[A-Za-z0-9.+]+)?$' | sort | uniq -c | sed 's/^/    /' || true
