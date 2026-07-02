#!/bin/bash
# A/B benchmark driver.
#
# A = a pristine checkout of BASE_REF (default HEAD) built in a temp worktree.
# B = the current working tree (your candidate change).
#
# Both are compiled to AOT executables and run interleaved (A B A B ...)
# from the repository root so they share dylibs, models, and machine state.
#
# Usage: tool/bench/run_ab.sh [rounds] [base_ref] [bench args passed to both]
# Example: tool/bench/run_ab.sh 6 HEAD --filter=cm_ --samples=60

set -euo pipefail

ROUNDS="${1:-6}"
BASE_REF="${2:-HEAD}"
if [ "$#" -ge 2 ]; then shift 2; elif [ "$#" -ge 1 ]; then shift 1; fi
BENCH_ARGS=("$@")

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
OUT_DIR="$REPO_ROOT/.dart_tool/bench"
mkdir -p "$OUT_DIR"
WORK="$(mktemp -d)"
trap 'git worktree remove --force "$WORK/base" 2>/dev/null || true; rm -rf "$WORK"' EXIT

echo "== compiling B (working tree) =="
dart compile exe tool/bench/bench.dart -o "$WORK/bench_b" >/dev/null

echo "== preparing A (worktree at $BASE_REF) =="
git worktree add --detach "$WORK/base" "$BASE_REF" >/dev/null 2>&1
(cd "$WORK/base" && { flutter pub get --offline >/dev/null 2>&1 || flutter pub get >/dev/null; })
(cd "$WORK/base" && dart compile exe tool/bench/bench.dart -o "$WORK/bench_a" >/dev/null)

A_OUT="$OUT_DIR/last_a.jsonl"
B_OUT="$OUT_DIR/last_b.jsonl"
LOG="$OUT_DIR/last_run.log"
: > "$A_OUT"
: > "$B_OUT"
: > "$LOG"

echo "== interleaved runs: $ROUNDS rounds (log: $LOG) =="
for i in $(seq 1 "$ROUNDS"); do
  echo "-- round $i/$ROUNDS --"
  caffeinate -dims "$WORK/bench_a" --out="$A_OUT" --label=A "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}" >>"$LOG" 2>&1
  caffeinate -dims "$WORK/bench_b" --out="$B_OUT" --label=B "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}" >>"$LOG" 2>&1
done

echo "== verdict (A=$BASE_REF, B=working tree) =="
dart run tool/bench/compare.dart "$A_OUT" "$B_OUT"
