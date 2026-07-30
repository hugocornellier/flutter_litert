#!/bin/bash
# Point the iOS SPM package at a locally built TensorFlowLiteCCoreML.xcframework.
#
# ios/flutter_litert/Package.swift resolves TensorFlowLiteCCoreML as a
# .binaryTarget(url:checksum:) from a GitHub release, so a locally built framework
# cannot be tested without redirecting it. This rewrites that one target to
# .binaryTarget(name:path:) and can put it back.
#
# SPM rejects absolute binaryTarget paths ("path expected to be relative to package
# root"), so the xcframework is copied into the package directory and referenced
# relatively. Both the copy and the Package.swift edit are undone by --revert.
#
# This is a TEST-ONLY local edit. Do not commit either change.
#
# Usage:
#   scripts/use_local_coreml_xcframework.sh /abs/path/TensorFlowLiteCCoreML.xcframework
#   scripts/use_local_coreml_xcframework.sh --revert
set -euo pipefail

PKG="$(cd "$(dirname "$0")/.." && pwd)/ios/flutter_litert/Package.swift"
MARKER="// LOCAL-COREML-OVERRIDE"

STAGED_NAME="TensorFlowLiteCCoreML.local.xcframework"

revert() {
  rm -rf "$(dirname "$PKG")/$STAGED_NAME"
  if ! grep -q "$MARKER" "$PKG"; then
    echo "no local override present; nothing to revert"
    exit 0
  fi
  git -C "$(dirname "$PKG")" checkout -- Package.swift 2>/dev/null \
    || git -C "$(cd "$(dirname "$0")/.." && pwd)" checkout -- ios/flutter_litert/Package.swift
  echo "reverted Package.swift to the committed version"
  grep -n "TensorFlowLiteCCoreML" "$PKG" | head -3
}

if [ "${1:-}" = "--revert" ]; then revert; exit 0; fi

XC="${1:-}"
if [ -z "$XC" ] || [ ! -d "$XC" ]; then
  echo "usage: $0 /abs/path/TensorFlowLiteCCoreML.xcframework | --revert" >&2
  exit 1
fi

if grep -q "$MARKER" "$PKG"; then
  echo "override already applied; run --revert first" >&2
  exit 1
fi

# SPM needs it inside the package dir, referenced relatively.
rm -rf "$(dirname "$PKG")/$STAGED_NAME"
cp -R "$XC" "$(dirname "$PKG")/$STAGED_NAME"
echo "staged $(basename "$XC") as $STAGED_NAME inside the package"

python3 - "$PKG" "$STAGED_NAME" "$MARKER" <<'PY'
import re, sys
pkg, xc, marker = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(pkg).read()
pat = re.compile(
    r'\.binaryTarget\(\s*\n\s*name:\s*"TensorFlowLiteCCoreML",\s*\n'
    r'\s*url:\s*"[^"]*",\s*\n\s*checksum:\s*"[^"]*"\s*\n\s*\)', re.M)
m = pat.search(src)
if not m:
    sys.exit("could not find the TensorFlowLiteCCoreML binaryTarget to rewrite")
repl = (f'        {marker}\n'
        f'        .binaryTarget(\n'
        f'            name: "TensorFlowLiteCCoreML",\n'
        f'            path: "{xc}"\n'
        f'        )')
open(pkg, "w").write(src[:m.start()] + repl.lstrip() + src[m.end():])
print("rewrote TensorFlowLiteCCoreML to a local path")
PY

grep -n -A3 "$MARKER" "$PKG"
echo
echo "Remember: run '$0 --revert' when finished."
