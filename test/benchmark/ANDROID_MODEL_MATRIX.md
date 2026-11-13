# Android physical-device model matrix

The Android matrix runs the same deterministic tensor-parity accuracy check as
the macOS matrix against all 29 published models. It measures every Android
backend currently exposed by `flutter_litert`:

- Interpreter: plain CPU and XNNPACK.
- CompiledModel: every non-empty combination of CPU, GPU, and NPU at fp32,
  plus strict GPU fp16.

That is 10 backend cells per model and 290 rows in the merged dataset. The
recorded 3.x datasets (`ANDROID_*MODEL_MATRIX_RESULTS.json`) also measured
`interpreter_flex` and `interpreter_gpu_v2_*`, which were removed in 4.0.0, so
the Galaxy S23 dataset has 13 cells per model and 377 rows. Each
successful cell executes three finite, non-degenerate fixtures and compares
every float output to the plain CPU Interpreter reference using
`atol = 1e-4` and `rtol = 1%`. Warm-up and timed samples were configured by the
workflow (5 and 15 by default).

## Firebase Test Lab execution

The recorded datasets came from the `android-model-matrix-testlab.yml`
workflow, which has since been removed from the repository; new runs are
started in Firebase Test Lab by hand. That workflow pinned the exact commits of
the seven public model repositories, balanced the 29 models into five shards,
and built one arm64 APK per shard. A free virtual CPU-only smoke had to pass
before it submitted the five physical runs. The `device_model` input picked the
device: `dm1q` (Galaxy S23, the default), `caiman` (Pixel 9 Pro), or `a56x`
(Galaxy A56). The `abi` input could switch the shards to the 32-bit
`armeabi-v7a` build. On the Galaxy S23 the physical device
supplies the Android GPU drivers and Snapdragon 8 Gen 2 HTP v73 NPU used by
every shard. The merged report names the device recorded in the shard
metadata.

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

Every workflow run uploaded one artifact containing:

- `ANDROID_MODEL_MATRIX_RESULTS.json`: full metadata, references, accuracy
  cases, timing distributions, errors, and all 290 rows;
- `ANDROID_MODEL_MATRIX_RESULTS.csv`: flattened rows for analysis;
- `ANDROID_MODEL_MATRIX_REPORT.md`: the giant 29-model × 10-backend table;
- `raw/`: Test Lab logcat, instrumentation result, and JUnit XML per shard.

**Note (2026-09-24): the `cat_detection` and `dog_detection` rows in the three
recorded Android datasets (`ANDROID_*MODEL_MATRIX_RESULTS.json`) are stale.**
They measured the models those packages shipped before 2026-08-12: dynamic-batch
exports (landmarks sha256 `b44c9495…` cat, `66409cca…` dog), pinned in the
matrix manifest at `cat_detection` `500d6ae` and `dog_detection` `98fde41`.
Their Compiled CPU failures (`LiteRtStatus=3`) match what LiteRT's Python
CompiledModel does on those same files. Since 3.0.1 both packages ship static
exports, which CompiledModel runs on CPU and GPU; see the 2026-09-24 update in
[`doc/graph_shape_vs_delegate.md`](../../doc/graph_shape_vs_delegate.md). Move
the two pins forward and re-run before quoting these rows.

The workflow deliberately failed its final collection gate if any cell lacked a
directly emitted row, but uploaded the rectangular merged dataset first so a
native crash remained diagnosable.
