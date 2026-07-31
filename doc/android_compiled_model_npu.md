# Android CompiledModel NPU

## Scope

This checkpoint adds Android's runtime plumbing, device-targeted distribution,
unsupported-device fallback, and physical gates for three Qualcomm
generations. It intentionally does not claim universal Android NPU support:
vendor libraries are SoC-specific, and each vendor/HTP generation needs its
own physical-device correctness matrix.

| Device | Firebase model | Qualcomm SoC | HTP | Android |
|---|---|---|---|---|
| Samsung Galaxy S23 | `dm1q` | Snapdragon 8 Gen 2 / SM8550 | v73 | API 35 |
| Samsung Galaxy S24 Ultra | `e3q` | Snapdragon 8 Gen 3 / SM8650 | v75 | API 34 |
| Samsung Galaxy S25 Ultra | `pa3q` | Snapdragon 8 Elite / SM8750 | v79 | API 35 |

All targets use LiteRT Next 2.1.6 and QAIRT 2.47.0.260601 JIT libraries.

The ordinary package retains Flutter's minimum SDK and remains multi-ABI. The
device-targeted NPU bundle opts into minSdk 31 and arm64 because the official
modules require Android 12+. NPU is activated only by an explicit
`Accelerator.npu` request in an app that received a matching runtime.
CPU/GPU-only apps have no new vendor binaries or runtime initialization.

## Runtime design

LiteRT JIT NPU compilation requires two environment options:

- `CompilerPluginLibraryDir`, which is scanned for
  `libLiteRtCompilerPlugin_*.so`;
- `DispatchLibraryDir`, which is scanned for `libLiteRtDispatch_*.so` and is
  also passed to the vendor runtime so it can load its dependencies.

`flutter_litert` creates a separate Android NPU environment so this remains
correct when an ordinary CPU or GPU CompiledModel was initialized first. It
resolves the extracted directory containing `libLiteRt.so` from
`/proc/self/maps`, supplies both options, and enables LiteRT's compiler cache in
the app's writable temporary/cache directory.

The resolver deliberately rejects APK-backed `base.apk!/lib/...` mappings.
LiteRT scans with filesystem directory APIs, so Qualcomm apps must set:

```kotlin
android {
  packaging {
    jniLibs {
      useLegacyPackaging = true
    }
  }
}
```

This mirrors LiteRT's own Android `BuiltinNpuAcceleratorProvider`, whose library
directory is `ApplicationInfo.nativeLibraryDir`, without requiring an
asynchronous Flutter platform channel. The synchronous resolver is compatible
with Dart worker isolates, but the physical gate currently validates the main
isolate only.

Apps targeting Android 12+ must also declare device-provided native libraries
before the platform linker exposes them. The plugin manifest mirrors LiteRT's
optional `uses-native-library` declaration for Qualcomm's `libcdsprpc.so`, so
consuming apps need no manifest change and remain installable on non-Qualcomm
devices.

## Runtime distribution

Production distribution should follow LiteRT's device-targeted Play Feature
Delivery layout. Unpack `litert_npu_runtime_libraries_jit.zip`, run its
Qualcomm fetch script, and add the conditional dynamic-feature modules to the
host Android app. Only the module matching the device should be installed;
putting several vendor dispatch libraries in one directory is ambiguous because
LiteRT selects the first matching dispatch library.

The example app provides complete reference wiring for the three physically
validated generations. Given a prepared archive root containing
`runtime_strings` and the official `qualcomm_runtime_v73`, `_v75`, and `_v79`
modules, build with:

```text
-PflutterLitert.qualcommNpuFeatureRoot=/absolute/path/to/prepared/runtime/root
```

That opt-in configuration:

- includes all three official dynamic features and the shared strings module;
- packages a device-targeting configuration that maps SM8550, SM8650, and
  SM8750 to v73, v75, and v79 respectively;
- enables device-group bundle splits with `other` as the default group;
- raises only that bundle to minSdk 31, filters it to arm64 for Test Lab, and
  extracts native libraries for LiteRT's filesystem scanners.

Bundletool 1.18.3 validation confirmed that each supported device spec selects
exactly one matching feature and that `other` selects no Qualcomm feature. The
[virtual fallback matrix](https://console.firebase.google.com/project/flutter-litert/testlab/histories/bh.f942c79f8598095d/matrices/5517724590966787869)
then passed against the real AAB install path with no dispatch library present.

For a fused local/Test Lab APK, set:

```properties
flutterLitert.qualcommNpuRuntimeDir=/absolute/path/to/qualcomm_runtime_v75/src/main/jni/arm64-v8a
```

The plugin's generated JNI source then copies exactly one arm64 Qualcomm set.
For JIT, the directory must contain:

```text
libLiteRtCompilerPlugin_Qualcomm.so
libLiteRtDispatch_Qualcomm.so
libQnnHtp.so
libQnnSystem.so
libQnnHtpV75Skel.so
libQnnHtpV75Stub.so
libQnnHtpPrepare.so
libQnnIr.so
libQnnSaver.so
```

The v75 filenames change together for another supported HTP generation. Gradle
accepts v69, v73, v75, v79, or v81, requires one matching Skel/Stub pair, and
rejects incomplete or competing LiteRT vendor libraries.

The QAIRT SDK archive is approximately 2.35 GB. Each delivered generation has
nine NPU files totaling about 105 MB; the prepared three-generation feature
root is about 315 MB, but Play delivers only one module. The manual CI workflow
downloads the official archives, verifies their SHA-256 digests, extracts only
those 27 files, and caches the prepared root. It never commits or republishes
the Qualcomm binaries.

## Placement and validation semantics

- `{Accelerator.npu}` is the strict hardware proof. If the compiler cannot
  translate the whole graph or dispatch cannot initialize, model construction
  must fail.
- `{Accelerator.npu, Accelerator.cpu}` allows partial delegation and CPU
  fallback by design. A successful build alone is therefore not proof that the
  NPU contributed.
- If device-targeted delivery installed no Android NPU runtime, strict `{npu}`
  throws an actionable `UnsupportedError`. A mixed request removes only NPU,
  continues with its explicitly requested GPU/CPU fallback, and reports the
  effective accelerator set on the resulting `CompiledModel`.
- `CompiledModel.isFullyAccelerated` proves that selected delegates covered the
  whole graph, but in mixed mode it cannot identify which delegate ran each
  operation.
- Validate fixed outputs against bare CPU with `verifyCompiledModel`, and
  validate application-level detections/landmarks separately before production
  rollout.

The manual workflow `.github/workflows/android-npu-testlab.yml` couples each
physical device with its required runtime generation through the `target`
input. It has two test scopes:

1. default smoke: packaged-library audit, CPU-first regression, strict NPU
   known-output inference, five repeated inferences, and three
   create/run/close cycles;
2. `full_sweep=true`: all smoke checks plus CPU-reference comparisons for
   MobileFaceNet, selfie multiclass segmentation, and the heavy pose-landmark
   model. The workflow records known tolerance rejections without discarding a
   successful runtime matrix; the test's default outside CI still enforces the
   1% gate.

The workflow is `workflow_dispatch` only. No push or pull request can
automatically consume the repository's five daily Firebase Test Lab runs.

As observed in this [AAB delivery matrix](https://console.firebase.google.com/project/flutter-litert/testlab/histories/bh.f942c79f8598095d/matrices/4699985412214320442)
on August 1, 2026, Test Lab accepts the bundle but installs its default device
group instead of evaluating the beta SoC condition. The workflow therefore
validates the production AAB first, then uses the official bundletool to derive
a universal APK containing `base` plus exactly the selected runtime module.
This preserves a meaningful hardware test without pretending Test Lab
validated Play's conditional-serving decision. Local bundletool extraction and
the virtual default-group matrix cover that decision separately.

## Physical validation

Validated on July 30 and August 1, 2026, using all three physical
configurations above. LiteRT logs confirmed that the Qualcomm compiler handled
the strict graph, `DispatchDelegate` replaced its only node, and
`QnnGraph_execute` completed successfully on every HTP generation. Times are
from debug Test Lab builds and are diagnostic, not benchmarks.

| Strict target | JIT compile | Warm inference | Result |
|---|---:|---:|---|
| Galaxy S23 / SM8550 / HTP v73 | 311 ms | approximately 2.5 ms | [pass](https://console.firebase.google.com/project/flutter-litert/testlab/histories/bh.f942c79f8598095d/matrices/6765465753715362641) |
| Galaxy S24 Ultra / SM8650 / HTP v75 | 247 ms | 2.6–3.1 ms | [pass](https://console.firebase.google.com/project/flutter-litert/testlab/histories/bh.f942c79f8598095d/matrices/5345184755930954190) |
| Galaxy S25 Ultra / SM8750 / HTP v79 | 291 ms | approximately 2.5 ms | [pass](https://console.firebase.google.com/project/flutter-litert/testlab/histories/bh.f942c79f8598095d/matrices/8606149841659679742) |

Each strict run passed five consecutive HTP inferences and three fresh
create/run/close cycles after a CPU CompiledModel had initialized first.

The representative sweeps fully accelerated every listed graph. Deviations are
the maximum absolute NPU-vs-CPU difference as a percentage of the CPU output
range:

| Target | MobileFaceNet | Selfie multiclass | Heavy pose |
|---|---:|---:|---:|
| S23 / v73 | 0.321% (pass) | 1.839% (reject) | 6.449% (reject) |
| S24 Ultra / v75 | 0.332% (pass) | 1.768% (reject) | 5.301% (reject) |
| S25 Ultra / v79 | 0.307% (pass) | 1.819% (reject) | 5.561% (reject) |

The S24 and S25 diagnostic matrices are green because they record accuracy
rejections after proving compilation and execution. The standalone test still
enforces `kDefaultBackendTolerance` by default. The earlier
[enforcing S23 sweep](https://console.firebase.google.com/project/flutter-litert/testlab/histories/bh.f942c79f8598095d/matrices/4890302984279070446)
is intentionally red at its final safety assertion. In every case,
MobileFaceNet met the conservative 1% tolerance while segmentation and heavy
pose did not; those two models require application-level validation before NPU
rollout.
