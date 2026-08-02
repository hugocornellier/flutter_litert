#!/usr/bin/env bash

# Exhaustive Apple published-model accuracy + latency matrix.
#
# Defaults to macOS. Pass --ios to drive a tethered iPhone instead; the two
# targets run the identical mode set and write separate datasets.
#
# Run from any directory. Optional environment overrides:
#   MODEL_REPOS_ROOT, MATRIX_MODEL_FILTER, MATRIX_MODE_FILTER,
#   MATRIX_ITERS, MATRIX_WARMUP, MATRIX_ENFORCE_ACCURACY,
#   APPLE_MATRIX_JSON, APPLE_MATRIX_CSV

SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIRECTORY/../.." && pwd)"

cd "$REPOSITORY_ROOT/example" || exit 1
exec dart run tool/run_apple_model_matrix.dart "$@"
