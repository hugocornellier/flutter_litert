# Apple published-model matrix

This integration suite exercises every published `.tflite` asset in the seven
sibling detection repositories through both `Interpreter` and `CompiledModel`.
It covers macOS and iOS, which run the identical mode set so their columns are
directly comparable.

Run it in profile mode:

```sh
test/benchmark/run_apple_model_matrix.sh          # macOS
test/benchmark/run_apple_model_matrix.sh --ios    # tethered iPhone
```

`--ios` auto-detects the connected physical device and refuses to guess when
more than one is attached; pass `--device=<udid>` to choose explicitly. The
host must be macOS in both cases, and its Xcode must be at least as new as the
device's iOS version or the developer disk image cannot mount.

For diagnosis or a smaller run, set comma-separated exact labels in the
`MATRIX_MODEL_FILTER=repository/model` and/or `MATRIX_MODE_FILTER=mode`
environment variables, or pass `--model-filter=` / `--mode-filter=` (these
options require `=`, not a space). The normal runner leaves both empty and runs
the complete manifest.

Each target overwrites its own pair of complete, machine-readable datasets,
`MACOS_MODEL_MATRIX_RESULTS.*` or `IOS_MODEL_MATRIX_RESULTS.*`:

- the `.json` contains run metadata, the 29-model inventory and SHA-256 hashes,
  CPU reference tensor metadata and signatures, detailed accuracy cases,
  backend evidence, timings, and full errors.
- the `.csv` is a flattened model × mode table suitable for spreadsheets, SQL,
  or plotting.

The latest authoritative macOS run is summarized in
[`MACOS_MODEL_MATRIX_REPORT.md`](MACOS_MODEL_MATRIX_REPORT.md).

## Model source

macOS reads the `.tflite` files straight from the sibling published checkouts
under `MODEL_REPOS_ROOT`. An iPhone is sandboxed and cannot reach them, so the
same models are staged into the app bundle first:

```sh
dart run example/tool/stage_android_model_matrix_assets.dart \
  --repositories-root <dir> \
  --asset-root <repo>/example/assets/models/model_matrix \
  --shard all
```

That tool pins each repository to the commit recorded in the matrix manifest,
so a staged bundle is byte-identical to what the macOS and Android runs used.
Only the byte source differs between targets; every downstream stage sees the
same bytes, which is what keeps the datasets comparable.

Each row carries `row_started_utc`. On a passively cooled phone this makes
thermal throttling visible as drift through the run rather than letting it
silently depress whichever modes happen to run last.

The host builds the profile app once, then runs each mode in a fresh process.
This is required because the regular Core ML Interpreter delegate and the
dedicated CompiledModel Core ML NPU delegate can collide when loaded into one
process. If any model triggers an uncaught native signal or C++ abort, the
host uses the last model/phase marker and native evidence to identify the exact
cell, reruns the unaffected prefix and suffix, and falls back to recursive
bisection if a process exit has no native evidence. It records the Apple `.ips`
exception, termination reason, faulting native frames, fatal C++ line, and a
bounded per-model native log excerpt, then continues. Recoverable Dart/LiteRT
errors retain their exception type, phase, message, symbolic status code, Dart
stack, and the corresponding native diagnostics directly.

The `.ips` half of that evidence is macOS-only: the host harvests
`~/Library/Logs/DiagnosticReports`, which holds the Mac's crash reports, not a
tethered phone's. iOS native terminations are therefore attributed from the
phase marker and the fatal native log line that `flutter drive` streams off the
device, without an accompanying crash report. That is the same mechanism that
classified two of the three macOS terminations in the recorded run.

Because native faults cannot be caught inside Dart, accuracy enforcement is
deferred until after the complete dataset is written. By default the command
then exits nonzero when the quality gate fails. Set
`MATRIX_ENFORCE_ACCURACY=false` to collect and inspect known failing cells while
still requiring a complete rectangular dataset.

## Coverage

Interpreter modes:

- plain CPU (four threads)
- XNNPACK (four threads)
- Metal with precision loss allowed (fp16-capable)
- Metal with precision loss disabled (fp32)
- Core ML with `AllDevices`
- the GL/CL `GpuDelegateV2` API (expected to report unsupported on both Apple
  platforms, where Metal is the GPU path; the row is kept so the Apple and
  Android tables stay column-comparable)
- Flex / Select TF Ops, with the optional `flutter_litert_flex` addon bundled
  into the matrix host (Flex is a compatibility delegate, not an accelerator)

CompiledModel modes:

- strict CPU
- strict GPU fp16 and fp32
- strict NPU fp32
- GPU + CPU fallback
- NPU + CPU fallback
- NPU + GPU, with and without CPU (explicit rows documenting the Apple-platform
  restriction that NPU and GPU cannot be combined)

Every successful CompiledModel configuration is checked through both `run` and
`runAsync`. Timings include cold compilation, first sync/async inference, and
warmed sync/async latency distributions. Interpreter warmed timing measures
`invoke()` only; CompiledModel timing measures `run`/`runAsync` with managed I/O,
and the `timing_scope` column makes that distinction explicit.

For delegates that report partitioning through their native log, the host also
records structured delegated/total node and partition counts. This matters for
Flex in particular: a builtin-only model can run successfully with the Flex
delegate loaded while reporting zero compatible Flex nodes; such a row proves
compatibility, not that Flex executed model operations.

## Accuracy contract

The first accuracy layer is deterministic CPU-reference tensor parity, not a
claim about real-world model quality. For each model, the suite creates an
independent plain-CPU Interpreter reference and selects three finite fixtures
from these candidates:

- a non-degenerate constant `0.5`
- a `[0.05, 0.95]` prime-period ramp
- a stride-scrambled `[0.1, 0.9]` pattern
- a reverse `[0.05, 0.95]` ramp
- a second stride-scrambled `[0.2, 0.8]` pattern

A candidate that is outside a model's meaningful domain can legitimately make
the CPU reference non-finite (for example, identical face landmarks create a
degenerate normalization). Such a candidate is recorded under
`rejected_fixtures` and replaced by the next candidate; backends are still
required to pass exactly three finite cases.

Every output tensor must retain its count and length, contain only finite
values, and satisfy:

```text
absolute error <= 1e-4 + 0.01 * max(reference range, reference magnitude)
```

Compact vector outputs also record top-1 agreement as a diagnostic. A future
task-level layer can add labeled images and metrics such as IoU, mAP, and
normalized keypoint distance without replacing this backend-regression layer.

Unsupported, rejected, or crashing backends do not disappear: every model ×
mode pair has exactly one row with a status, failure phase, exception type, and
error message. Recoverable Interpreter errors include `TfLiteStatus` number and
name; CompiledModel errors include `LiteRtStatus` number and name. Native
signals and uncaught C++ exceptions are process-level failures and therefore
carry `.ips`/fatal-log evidence rather than pretending they were catchable Dart
exceptions. The final quality gate fails when a CPU reference cannot be
produced, a backend executes but violates the accuracy contract, an unexpected
execution/native failure occurs, or the table is not rectangular. Expected
unsupported combinations remain valid dataset rows.
