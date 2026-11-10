#!/usr/bin/env bash
# Verify a freshly built libtensorflowlite_flex-win.dll against the file it
# replaces (inspected only, never loaded). Runs in Git Bash with the MSVC
# developer environment on PATH (dumpbin, cl).
#
#   verify_windows.sh <new.dll> <old.dll> <out_dir>
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"

new="$1"
old="$2"
out="$3"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out"
report="$out/verify-report.txt"
: > "$report"
fail=0
log() { echo "$*" | tee -a "$report"; }
win() { cygpath -w "$1"; }

exports() {
  dumpbin /nologo /exports "$(win "$1")" | tr -d '\r' \
    | awk '/ordinal +hint +RVA +name/ { on = 1; next }
           on && /^ *Summary/ { on = 0 }
           on && NF >= 4 && $1 ~ /^[0-9]+$/ { print $4 }
           on && NF == 3 && $1 ~ /^[0-9]+$/ { print $3 }' | sort -u
}
versions() {
  grep -a -oE '(^|[^0-9.])2\.(19|20|21|22)\.[0-9]+(-[A-Za-z0-9.+]+)?' "$1" \
    | sed -E 's/^[^0-9]//' | sort | uniq -c | sort -rn | head -n 8
}

for f in "$new" "$old"; do
  log "== $(basename "$f") ($([ "$f" = "$new" ] && echo new || echo old)) =="
  log "size: $(wc -c < "$f") bytes"
  log "sha256: $(sha256sum "$f" | cut -d' ' -f1)"
  dumpbin /nologo /headers "$(win "$f")" | tr -d '\r' > "$out/headers.tmp"
  log "machine: $(grep -m1 'machine (' "$out/headers.tmp" | sed 's/^ *//')"
  log "OS version: $(grep -m1 'operating system version' "$out/headers.tmp" | awk '{print $1}')"
  log "subsystem version: $(grep -m1 'subsystem version' "$out/headers.tmp" | awk '{print $1}')"
  log "linker version: $(grep -m1 'linker version' "$out/headers.tmp" | awk '{print $1}')"
  log "dependents: $(dumpbin /nologo /dependents "$(win "$f")" | tr -d '\r' | grep -iE '^ +[^ ]+\.dll$' | sed 's/^ *//' | tr '\n' ' ')"
  log "version-like strings (count value):"
  versions "$f" | sed 's/^/  /' | tee -a "$report"
  log ""
done
rm -f "$out/headers.tmp"

# C/C++ runtime DLLs beyond the set the 2.20.0 DLL and the VC++ 2015-2022
# redistributable baseline cover. vcruntime140_threads.dll (C11 threads,
# VS 2022 17.8+) is only in recent redistributables, so call it out.
log "== runtime DLLs (new) =="
dumpbin /nologo /dependents "$(win "$new")" | tr -d '\r' | grep -iE '^ +[^ ]+\.dll$' \
  | sed 's/^ *//' | tr 'A-Z' 'a-z' > "$out/deps-new.txt"
extra_crt="$(grep -E '^(vcruntime|msvcp|ucrtbase|concrt|vcomp|api-ms-win-crt)' "$out/deps-new.txt" \
  | grep -vxE 'msvcp140\.dll|vcruntime140\.dll|vcruntime140_1\.dll|api-ms-win-crt-[a-z0-9-]+\.dll' || true)"
if grep -qx 'vcruntime140_threads.dll' "$out/deps-new.txt"; then
  log "vcruntime140_threads.dll: IMPORTED"
else
  log "vcruntime140_threads.dll: not imported"
fi
log "CRT DLLs beyond msvcp140, vcruntime140, vcruntime140_1 and api-ms-win-crt-*: ${extra_crt:-none}"
log ""

exports "$new" > "$out/exports-new.txt"
exports "$old" > "$out/exports-old.txt"
comm -13 "$out/exports-old.txt" "$out/exports-new.txt" > "$out/exports-added.txt"
comm -23 "$out/exports-old.txt" "$out/exports-new.txt" > "$out/exports-removed.txt"
log "== exports =="
log "old: $(wc -l < "$out/exports-old.txt")  new: $(wc -l < "$out/exports-new.txt")"
log "added: $(wc -l < "$out/exports-added.txt")  removed: $(wc -l < "$out/exports-removed.txt")"
log "first 40 added:"
head -n 40 "$out/exports-added.txt" | sed 's/^/  + /' | tee -a "$report"
log "first 40 removed:"
head -n 40 "$out/exports-removed.txt" | sed 's/^/  - /' | tee -a "$report"
for s in tflite_plugin_create_delegate tflite_plugin_destroy_delegate; do
  if grep -qx "$s" "$out/exports-new.txt"; then
    log "export $s: present"
  else
    log "FAIL: $s is not exported"
    fail=1
  fi
done
log ""

log "== smoke test (new only) =="
( cd "$out" && cl /nologo /O1 "$(win "$here/smoke_test.c")" /Fe:smoke_test.exe > cl.log 2>&1 ) \
  || { cat "$out/cl.log"; log "FAIL: could not compile smoke test"; fail=1; }
if [ -f "$out/smoke_test.exe" ]; then
  smoke="$("$out/smoke_test.exe" "$(win "$new")" 2>&1 | tr -d '\r')"
  log "$smoke"
  echo "$smoke" | grep -q '^SMOKE OK$' || { log "FAIL: smoke test"; fail=1; }
fi
rm -f "$out"/smoke_test.* "$out/cl.log"

gzip -f "$out/exports-new.txt" "$out/exports-old.txt"
log ""
log "verify result: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
