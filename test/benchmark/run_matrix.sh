#!/usr/bin/env bash
# Runs the engine x accelerator matrix in PROFILE mode (required for
# representative numbers) and appends long-format results to
# test/benchmark/RESULTS.csv.
#
# Usage: test/benchmark/run_matrix.sh [device]   # macos | linux | windows | <device-id>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEVICE="${1:-macos}"
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
export MATRIX_CSV="$REPO_ROOT/test/benchmark/RESULTS.csv"

cd "$REPO_ROOT/example"
flutter drive \
  --profile \
  --driver=test_driver/integration_test.dart \
  --target=integration_test/engine_matrix_test.dart \
  -d "$DEVICE" \
  --dart-define=LITERT_COMMIT="$COMMIT"

echo "Done. Appended results to $MATRIX_CSV"
