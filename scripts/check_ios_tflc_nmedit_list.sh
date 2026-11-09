#!/usr/bin/env bash
# Checks that the symbol hide list in ios/flutter_litert.podspec (the symbols
# TensorFlowLiteC shares with TensorFlowLiteFlex, which the podspec makes local
# with `xcrun nmedit -R` when flutter_litert_flex is present) still matches a
# TensorFlowLiteC.xcframework.
#
# Usage: scripts/check_ios_tflc_nmedit_list.sh <TensorFlowLiteC.xcframework> <podspec>
#
# For each slice the podspec processes (ios-arm64, ios-arm64_x86_64-simulator)
# and each architecture:
#   - every listed symbol must be a global defined symbol (else FAIL);
#   - `nmedit -R` on a copy must succeed and leave none of them global, with
#     the global count dropping by exactly the list length (else FAIL);
#   - exported XNNPack delegate / AcquireFlexDelegate symbols that are NOT on
#     the list are printed (not a failure: whether they overlap TensorFlowLiteFlex
#     depends on the flex build).
# Never runs the binary.
set -euo pipefail

xcfw="$1"
podspec="$2"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail=0

# The list is the heredoc between `<<~SYMS` and the closing `SYMS`.
awk '/<<~SYMS/ {f=1; next} f && /^[[:space:]]*SYMS[[:space:]]*$/ {f=0} f {gsub(/[[:space:]]/, ""); if ($0 != "") print}' \
  "$podspec" | sort -u > "$work/list"
n="$(wc -l < "$work/list" | tr -d ' ')"
echo "podspec hide list: $n symbols"
[ "$n" -gt 0 ] || { echo "FAIL: could not read the list from $podspec"; exit 1; }

for slice in ios-arm64 ios-arm64_x86_64-simulator; do
  bin="$xcfw/$slice/TensorFlowLiteC.framework/TensorFlowLiteC"
  if [ ! -f "$bin" ]; then
    echo "FAIL: missing $bin"
    fail=1
    continue
  fi
  cp "$bin" "$work/bin"
  for arch in $(lipo -archs "$bin"); do
    nm -arch "$arch" -gU "$bin" | awk 'NF>=3 {print $3}' | sort -u > "$work/before"
    missing="$(comm -23 "$work/list" "$work/before")"
    extra="$(grep -E '^_TfLiteXNNPackDelegate|AcquireFlexDelegate' "$work/before" | comm -23 - "$work/list" || true)"
    echo "=== $slice [$arch]: $(wc -l < "$work/before" | tr -d ' ') globals"
    if [ -n "$missing" ]; then
      echo "FAIL: listed but not exported:"
      printf '%s\n' "$missing" | sed 's/^/    /'
      fail=1
    else
      echo "all $n listed symbols are exported"
    fi
    if [ -n "$extra" ]; then
      echo "exported XNNPack/AcquireFlex symbols NOT on the list:"
      printf '%s\n' "$extra" | sed 's/^/    /'
    else
      echo "no other XNNPack delegate or AcquireFlexDelegate exports"
    fi
  done
  if ! xcrun nmedit -R "$work/list" "$work/bin" > "$work/nmedit.log" 2>&1; then
    echo "FAIL: nmedit -R exited non-zero:"
    sed 's/^/    /' "$work/nmedit.log"
    fail=1
    continue
  fi
  [ -s "$work/nmedit.log" ] && sed 's/^/    nmedit: /' "$work/nmedit.log"
  for arch in $(lipo -archs "$bin"); do
    nm -arch "$arch" -gU "$bin" | awk 'NF>=3 {print $3}' | sort -u > "$work/before"
    nm -arch "$arch" -gU "$work/bin" | awk 'NF>=3 {print $3}' | sort -u > "$work/after"
    b="$(wc -l < "$work/before" | tr -d ' ')"
    a="$(wc -l < "$work/after" | tr -d ' ')"
    still="$(comm -12 "$work/list" "$work/after")"
    echo "after nmedit $slice [$arch]: globals $b -> $a"
    if [ -n "$still" ] || [ $((b - a)) -ne "$n" ]; then
      echo "FAIL: nmedit did not hide exactly the $n listed symbols"
      fail=1
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo "RESULT nmedit list: FAIL"
  exit 1
fi
echo "RESULT nmedit list: OK (all $n symbols exported and hidden by nmedit -R)"
