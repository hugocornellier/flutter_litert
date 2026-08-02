# Android physical-device model matrix

The Android matrix runs the same deterministic tensor-parity accuracy check as
the macOS matrix against all 29 published models. It measures every Android
backend currently exposed by `flutter_litert`:

- Interpreter: plain CPU, XNNPACK, Flex, and GPU v2 GL/CL at fp16 and fp32.
- CompiledModel: every non-empty combination of CPU, GPU, and NPU at fp32,
  plus strict GPU fp16.

That is 13 backend cells per model and 377 rows in the merged dataset. Each
successful cell executes three finite, non-degenerate fixtures and compares
every float output to the plain CPU Interpreter reference using
`atol = 1e-4` and `rtol = 1%`. Warm-up and timed samples are configured by the
workflow (5 and 15 by default).

## Firebase Test Lab execution

[`android-model-matrix-testlab.yml`](../../.github/workflows/android-model-matrix-testlab.yml)
pins the exact commits of the seven public model repositories, balances the 29
models into five shards, and builds one arm64 APK per shard. A free virtual
CPU-only smoke must pass before it submits the five physical Galaxy S23 runs.
The physical device supplies the Android GPU drivers and Snapdragon 8 Gen 2
HTP v73 NPU used by every shard.

No backend exception aborts a shard. The row records the model, API, mode,
phase, Dart error type, stack, and any parsed numeric/symbolic `LiteRtStatus`
or `TfLiteStatus`. A hard native process crash cannot be caught by Dart, so
the harness writes phase and chunked JSON records to logcat immediately. The
host merger preserves completed cells and synthesizes explicit
`native_crash`/`not_executed_after_native_crash` rows for anything interrupted.

CompiledModel accelerator modes use synchronous `run` for the comparable
accuracy and latency result. Only CPU also exercises `runAsync`; Android GL/CL
drivers are thread-affine and the package documents mobile-accelerator
`runAsync` as unvalidated.

## Outputs

Every workflow run uploads one artifact containing:

- `ANDROID_MODEL_MATRIX_RESULTS.json`: full metadata, references, accuracy
  cases, timing distributions, errors, and all 377 rows;
- `ANDROID_MODEL_MATRIX_RESULTS.csv`: flattened rows for analysis;
- `ANDROID_MODEL_MATRIX_REPORT.md`: the giant 29-model × 13-backend table;
- `raw/`: Test Lab logcat, instrumentation result, and JUnit XML per shard.

The workflow deliberately fails its final collection gate if any cell lacks a
directly emitted row, but uploads the rectangular merged dataset first so a
native crash remains diagnosable.
