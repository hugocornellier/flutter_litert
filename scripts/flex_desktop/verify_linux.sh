#!/usr/bin/env bash
# Verify a freshly built libtensorflowlite_flex-linux.so against the file it
# replaces (inspected only, never loaded).
#
#   verify_linux.sh <new.so> <old.so> <out_dir>
#
# Writes <out_dir>/verify-report.txt plus full export lists, and fails if the
# plugin symbols are missing or the smoke test does not pass.
set -uo pipefail

new="$1"
old="$2"
out="$3"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out"
report="$out/verify-report.txt"
: > "$report"
fail=0
log() { echo "$*" | tee -a "$report"; }

floor() { objdump -T "$1" | grep -oE "$2_[0-9]+(\.[0-9]+)*" | sort -uV | tail -n 1; }
exports() { nm -D --defined-only "$1" | awk 'NF >= 3 { print $NF }' | sort -u; }
versions() {
  grep -a -oE '(^|[^0-9.])2\.(19|20|21|22)\.[0-9]+(-[A-Za-z0-9.+]+)?' "$1" \
    | sed -E 's/^[^0-9]//' | sort | uniq -c | sort -rn | head -n 8
}

for f in "$new" "$old"; do
  log "== $(basename "$f") ($([ "$f" = "$new" ] && echo new || echo old)) =="
  log "size: $(wc -c < "$f") bytes"
  log "sha256: $(sha256sum "$f" | cut -d' ' -f1)"
  log "file: $(file -b "$f")"
  log "machine: $(readelf -h "$f" | awk -F: '/Machine/ { gsub(/^ +/, "", $2); print $2 }')"
  log "NEEDED: $(readelf -d "$f" | grep NEEDED | grep -oE '\[[^]]+\]' | tr -d '[]' | tr '\n' ' ')"
  log "GLIBC floor: $(floor "$f" GLIBC)"
  log "GLIBCXX floor: $(floor "$f" GLIBCXX)"
  log "CXXABI floor: $(floor "$f" CXXABI)"
  log "version-like strings (count value):"
  versions "$f" | sed 's/^/  /' | tee -a "$report"
  log ""
done

exports "$new" > "$out/exports-new.txt"
exports "$old" > "$out/exports-old.txt"
comm -13 "$out/exports-old.txt" "$out/exports-new.txt" > "$out/exports-added.txt"
comm -23 "$out/exports-old.txt" "$out/exports-new.txt" > "$out/exports-removed.txt"
log "== exported (defined, dynamic) symbols =="
log "old: $(wc -l < "$out/exports-old.txt")  new: $(wc -l < "$out/exports-new.txt")"
log "added: $(wc -l < "$out/exports-added.txt")  removed: $(wc -l < "$out/exports-removed.txt")"
log "first 40 added (demangled):"
head -n 40 "$out/exports-added.txt" | c++filt | sed 's/^/  + /' | tee -a "$report"
log "first 40 removed (demangled):"
head -n 40 "$out/exports-removed.txt" | c++filt | sed 's/^/  - /' | tee -a "$report"
log "TfLite C API exports, new: $(grep -c '^TfLite' "$out/exports-new.txt")  old: $(grep -c '^TfLite' "$out/exports-old.txt")"
log ""

log "== plugin symbols in new =="
plugin="$(nm -D --defined-only "$new" | grep -E ' tflite_plugin_(create|destroy)_delegate$')"
log "$plugin"
for s in tflite_plugin_create_delegate tflite_plugin_destroy_delegate; do
  if ! echo "$plugin" | grep -qE " [TW] $s\$"; then
    log "FAIL: $s is not exported as a defined function"
    fail=1
  fi
done
log ""

log "== smoke test (new only) =="
gcc -O1 -o "$out/smoke_test" "$here/smoke_test.c" -ldl
smoke="$("$out/smoke_test" "$new" 2>&1)"
log "$smoke"
echo "$smoke" | grep -q '^SMOKE OK$' || { log "FAIL: smoke test"; fail=1; }

gzip -f "$out/exports-new.txt" "$out/exports-old.txt"
rm -f "$out/smoke_test"
log ""
log "verify result: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
