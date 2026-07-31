# Android NPU Multi-Vendor Delivery Design

Status: DRAFT (research and implementation plan; not yet implemented)

Research date: 2026-08-01

Related implementation checkpoint:
[Android CompiledModel NPU](android_compiled_model_npu.md)

## 1. Decision summary

`flutter_litert` should own Android NPU vendor integration. Detection packages
such as `face_detection_tflite` should not download, select, package, or load
Qualcomm, MediaTek, Samsung, or Google Tensor libraries themselves.

The design must support two delivery modes from the same vendor-pack registry:

1. **Embedded delivery for standalone APKs.** Every vendor selected by the app
   is signed into one APK. At runtime, `flutter_litert` detects the phone's SoC,
   prepares only the matching pack, and falls back to GPU or CPU when no usable
   NPU pack exists.
2. **Google Play delivery for Android App Bundles.** Each vendor or SoC runtime
   is a device-targeted Play Feature Delivery module. Google Play sends only
   the module matching the user's SoC. AOT models can use Play AI Packs.

The runtime selection, validation, fallback, diagnostics, and public Dart API
remain the same in both modes. Only the source of the selected files changes.

```text
                         flutter_litert vendor registry
                                      |
                              detect the phone SoC
                                      |
                             select one vendor pack
                                      |
                   +------------------+------------------+
                   |                                     |
          Standalone APK source                    Google Play source
          pack is in signed APK                pack is an installed split
                   |                                     |
                   +------------------+------------------+
                                      |
                    isolated LiteRT compiler/dispatch path
                                      |
                            NPU -> GPU -> CPU policy
```

This design deliberately avoids downloading native executable files from a
GitHub release when the installed app first runs. That would weaken offline
operation and integrity guarantees, complicate vendor licences, and violate
Google Play policy for apps distributed through Play.

## 2. Scope and terminology

In this document, **vendor** means a LiteRT CompiledModel NPU backend vendor.
It does not mean every Android GPU driver or the ordinary LiteRT CPU backend.

The desired product behavior is:

- one standalone APK can cover all NPU vendors the app author enables;
- a Play consumer can publish one AAB and avoid sending every NPU runtime to
  every phone;
- higher-level detection packages share the one `flutter_litert` installation
  and do not duplicate native runtime files;
- new vendors are added through a pack descriptor and adapter, rather than
  vendor-specific branches throughout the Dart and native code;
- unsupported phones still work through GPU and/or CPU fallback;
- NPU use is observable and model-specific accuracy remains gated.

"All Android phones" cannot mean that every Android phone runs on an NPU.
LiteRT NPU support currently requires Android API 31 or newer, arm64, a
supported SoC, usable vendor system drivers, and a compatible model. The
portable promise should instead be:

> Use a validated NPU when available; otherwise continue safely with the
> explicitly allowed GPU or CPU backend.

## 3. Current LiteRT NPU vendor support

LiteRT currently documents five NPU vendors. Four are relevant to Android.

| Vendor | Android | Compilation | Documented devices or SoCs | Packaging status observed in LiteRT 2.1.6 |
|---|---:|---|---|---|
| Qualcomm AI Engine Direct | Yes | JIT and AOT | SM8450, SM8475, SM8550, SM8650, SM8750, SM8850 | Public runtime templates and fetch tooling are present |
| MediaTek NeuroPilot | Yes | JIT and AOT | Dimensity 7300, 8300, 9000, 9200, 9300, 9400, and 9500 families listed by LiteRT | Backend source exists, but no ready MediaTek module was present in the inspected public runtime archive |
| Samsung Exynos AI LiteCore | Yes | JIT and AOT | Exynos 2500 and 2600 | Backend source exists, but no ready Samsung module was present in the inspected public runtime archive |
| Google Tensor | Yes | AOT only; Beta | Tensor G5 / Pixel 10 family | Runtime template is present, but ordinary raw TFLite models cannot currently be JIT compiled on the phone |
| Intel OpenVINO | No for Android | JIT and AOT | Intel Core Ultra on Linux and Windows | Keep outside the Android APK design |

Official references:

- [LiteRT NPU overview](https://developers.google.com/edge/litert/next/npu)
- [Qualcomm backend](https://developers.google.com/edge/litert/next/qualcomm)
- [MediaTek backend](https://developers.google.com/edge/litert/next/mediatek)
- [Samsung backend](https://developers.google.com/edge/litert/next/samsung)
- [Google Tensor backend](https://developers.google.com/edge/litert/next/tensor-sdk)
- [Intel backend](https://developers.google.com/edge/litert/next/intel)

### 3.1 Documentation versus ready-to-package artifacts

The documentation describes the overall backend support planned and present in
LiteRT. That is not the same as a turnkey Android pack in the latest release.

The inspected [LiteRT v2.1.6 release](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.1.6)
was published on 2026-07-02. Its two public NPU archives were:

- `litert_npu_runtime_libraries_jit.zip`: 2,757,150 bytes;
- `litert_npu_runtime_libraries.zip`: 982,686 bytes.

Although the current NPU guide shows example module names for Qualcomm,
MediaTek, Samsung, and Google Tensor, the downloaded v2.1.6 archives contained
only Qualcomm and Google Tensor module templates. MediaTek and Samsung source
trees are in LiteRT, but `flutter_litert` would currently have to build their
compiler/dispatch plugins and legally source any required SDK libraries.

This distinction must be shown in our support status:

- **Documented by LiteRT** is an upstream capability statement.
- **Packaged by flutter_litert** means a reproducible pack can be built.
- **Validated by flutter_litert** means strict hardware execution and model
  correctness were demonstrated on physical hardware.
- **Ready** requires all three plus an acceptable licence and fallback story.

## 4. What the Qualcomm work proved

Qualcomm is the first implemented Android vendor. Physical Firebase Test Lab
runs validated LiteRT Next 2.1.6 with QAIRT 2.47.0.260601 on:

| Device | SoC | HTP generation | Strict NPU result |
|---|---|---|---|
| Galaxy S23 | Snapdragon 8 Gen 2 / SM8550 | v73 | Passed |
| Galaxy S24 Ultra | Snapdragon 8 Gen 3 / SM8650 | v75 | Passed |
| Galaxy S25 Ultra | Snapdragon 8 Elite / SM8750 | v79 | Passed |

The logs proved that the Qualcomm compiler accepted the strict test graph, the
dispatch delegate replaced it, and QNN executed it. Each target also passed
five repeated inferences and three fresh create/run/close cycles after a CPU
CompiledModel had initialized first.

The tested accelerators were Qualcomm HTP NPU and CPU reference/fallback. The
strict `{npu}` configuration was the hardware proof. Mixed NPU plus CPU tested
fallback behavior and CPU supplied the correctness reference. These runs did
not make a new cross-vendor GPU hardware claim.

### 4.1 Runtime readiness is not model readiness

All three representative models fully accelerated, but NPU and CPU output
differences were model-dependent:

| Model | S23 / v73 | S24 Ultra / v75 | S25 Ultra / v79 | 1% gate |
|---|---:|---:|---:|---|
| MobileFaceNet | 0.321% | 0.332% | 0.307% | Passed |
| Selfie multiclass segmentation | 1.839% | 1.768% | 1.819% | Rejected |
| Heavy pose landmarks | 6.449% | 5.301% | 5.561% | Rejected |

Therefore, a successfully loaded NPU runtime is not permission to silently move
every model to NPU. Each production model or model family needs a CPU golden
comparison and an application-level quality check.

### 4.2 Qualcomm status

The Qualcomm runtime path is ready for the three tested SoCs, subject to
model-specific validation. Qualcomm as an entire vendor is not yet universally
validated. LiteRT also lists SM8450, SM8475, and SM8850, which were not covered
by these physical runs. Their corresponding runtime generations still need the
same strict hardware and correctness gates.

Detailed Test Lab links, logs, bundle behavior, and runtime file requirements
are recorded in [Android CompiledModel NPU](android_compiled_model_npu.md).

## 5. Ownership boundaries

### 5.1 What `flutter_litert` owns

`flutter_litert` should own:

- vendor and SoC detection;
- the vendor registry and pack schema;
- build-time artifact acquisition and hash verification;
- standalone APK and Play AAB packaging adapters;
- runtime availability checks and pack preparation;
- isolated compiler and dispatch directories;
- LiteRT environment options and compilation cache setup;
- fallback policy and effective-backend diagnostics;
- model compatibility/validation metadata;
- shared tests and physical-device readiness records.

### 5.2 What consuming apps own

The consuming app should make only app-level product choices:

- which ready vendors to include;
- embedded APK or Play delivery;
- whether NPU is strict or may fall back;
- whether a particular model is approved for NPU;
- any required vendor SDK licence acceptance.

### 5.3 What detection packages own

Packages such as `face_detection_tflite` should own their model semantics and
quality gates. They should not contain vendor `.so` files or vendor selectors.
They should request an accelerator policy through a shared `flutter_litert`
session API, or inherit the application's process-wide policy.

All detection packages in an app resolve one version of `flutter_litert`, so
native vendor packs are not copied once per dependency. The final Android app
packages them once.

## 6. Build outputs and delivery modes

### 6.1 Proposed consumer configuration

The exact Gradle/Dart syntax is an implementation detail, but the public choice
should be no more complex than this conceptual configuration:

```kotlin
flutterLitert {
  androidNpu {
    vendors = listOf("qualcomm", "mediatek") // or "allReady"
    delivery = "embedded"                    // "embedded", "play", or "none"
  }
}
```

Defaults should remain conservative:

- `delivery = "none"` unless an app opts in;
- no vendor SDK download for ordinary CPU/GPU consumers;
- no package-wide minimum Android or ABI change unless NPU packaging requires
  it and the app explicitly selected it;
- fallback behavior remains explicit.

### 6.2 Standalone public APK: embedded delivery

This is the primary mode for Agelapse.

The APK contains the compressed pack data for every enabled vendor. Android
cannot inspect the SoC after downloading a universal APK and remove unrelated
bytes from that already downloaded APK. Consequently, a Qualcomm user still
downloads the MediaTek and Samsung compressed packs if all three were enabled.

At runtime, only the matching pack is prepared for LiteRT. This limits active
runtime conflicts and additional extracted storage, but it does not reduce the
APK download itself.

```text
agelapse-release.apk
  assets/flutter_litert/npu/v1/
    manifest.json
    blobs/<sha256>...
    packs/qualcomm.json
    packs/mediatek.json
    packs/samsung.json
```

Executable libraries should be inside the APK before signing. No network is
required on first NPU use.

An early implementation gate must verify Android linker behavior for loading
the selected libraries from an app-private prepared directory. If a target
Android version or vendor dependency cannot load safely from that directory,
the build adapter must use an alternative signed-APK layout while preserving
one isolated LiteRT scan directory. We must not assume that arbitrary asset
extraction is executable on every supported Android release without a physical
test.

Possible app products remain available:

- one larger `allReady` universal APK;
- a smaller single-vendor APK for a controlled fleet;
- CPU/GPU-only APK with no vendor pack.

The universal multi-vendor APK is the default recommendation for public
standalone releases that want the widest supported coverage.

### 6.3 Google Play AAB: device-targeted delivery

An Android App Bundle is an upload format, not the one APK installed on every
device. Google Play produces and installs a base APK plus matching split APKs.

Google Play's device targeting can match the system-on-chip manufacturer and
model on API 31 or newer. This is currently a Beta Android/Play capability.
Each NPU pack or Qualcomm HTP generation can therefore be a conditional dynamic
feature:

```text
uploaded app.aab
  base
  litert_npu_qualcomm_v69
  litert_npu_qualcomm_v73
  litert_npu_qualcomm_v75
  litert_npu_qualcomm_v79
  litert_npu_qualcomm_v81
  litert_npu_mediatek
  litert_npu_samsung
  litert_npu_google_tensor
```

A supported Qualcomm device receives the base plus one matching Qualcomm
module. A MediaTek device receives the base plus its MediaTek module. An
unsupported device receives the base and uses GPU/CPU fallback. This avoids
charging each user for every vendor pack.

References:

- [Device targeting](https://developer.android.com/google/play/device-targeting)
- [Conditional Play Feature Delivery](https://developer.android.com/guide/playcore/feature-delivery/conditional)
- [LiteRT NPU runtime deployment](https://developers.google.com/edge/litert/next/npu)

The recommended Play policy is **conditional install-time delivery**. The pack
is ready when the app starts, inference remains available offline after
installation, and initialization does not need a download user interface.

An optional **on-demand** mode may be added later. It could reduce the initial
download further, but initialization becomes asynchronous and must handle
network failures, progress, user confirmation for large modules, cancellation,
and fallback while the module is absent. Play Feature Delivery, rather than a
`flutter_litert` or GitHub server, must perform that executable-code download.

Google Play policy states that Play-distributed apps may not download native
executable code such as `.so` files from outside Google Play:
[Device and Network Abuse policy](https://support.google.com/googleplay/android-developer/answer/16559646).

### 6.4 Dynamic features must integrate with the app project

Android dynamic-feature modules depend on the base application module and must
be known during Gradle settings/configuration. An AAR hidden inside a normal
Flutter plugin cannot independently become the app's device-targeted feature
split.

`flutter_litert` should still own the integration. The implementation spike
should choose between:

- a small settings-level Gradle plugin supplied by `flutter_litert`; or
- a `dart run flutter_litert:configure_android_npu` command that generates and
  updates clearly marked modules in the consuming app.

The acceptance requirement is one app-level opt-in/configuration, not copied
vendor code. Generated modules must remain reproducible and safe to regenerate.

### 6.5 A universal APK generated from an AAB

A `bundletool` universal APK is useful for local tests, Firebase Test Lab, and
some sideload workflows. It must not be assumed to have the same contents as
the recommended standalone embedded build. Feature fusing and the configured
default device group determine what is included.

For a public standalone release, build the explicit `embedded` APK product.
For a Play release, build the `play` AAB product. Use `bundletool` outputs to
test Play packaging, not as an accidental replacement for the embedded design.

### 6.6 AOT models and Play AI Packs

Google Tensor currently requires AOT-compiled models. A small Tensor runtime
pack alone is insufficient: every supported model needs a Tensor-compatible
compiled artifact.

For Play consumers, those artifacts can use device-targeted
[Play for On-device AI](https://developer.android.com/google/play/on-device-ai)
AI Packs. For standalone APK consumers, the corresponding AOT model variants
must be signed into the APK or handled by a separate, legally compliant model
delivery design.

This is why Google Tensor should follow the JIT-capable MediaTek and Samsung
work unless upstream Tensor JIT becomes available.

## 7. Vendor-pack architecture

### 7.1 Manifest-driven registry

The runtime should not contain an expanding central `switch` over vendors.
Each vendor pack should implement or describe the same contract:

```text
NpuVendorPack
  schemaVersion
  packId
  vendorId
  vendorSdkVersion
  liteRtVersion
  supportedAbis
  minimumAndroidApi
  compilationModes        // JIT, AOT, or both
  supportedSoCs           // exact manufacturer/model pairs
  requiredSystemLibraries
  compilerPluginFiles
  dispatchPluginFiles
  dependencyFiles
  commonBlobs
  socOverlayBlobs
  sourceAndLicenceMetadata
  sha256AndSizeForEveryBlob
  validationStatus
  testedDevicesAndModels
```

Adding a vendor then consists of:

1. a descriptor;
2. a build/fetch recipe;
3. a probe/preparation adapter when generic behavior is insufficient;
4. a device and model test matrix.

The general selection, extraction, fallback, and diagnostics code does not
change.

### 7.2 Content-addressed pack storage

Store APK blobs by SHA-256 and let pack manifests reference them. This permits
exact verification and deduplicates files shared by several runtime variants.

This matters especially for Qualcomm. The current official generation modules
repeat large common QNN files. The final APK should contain one common base and
small v69/v73/v75/v79/v81 overlays wherever licences and loader behavior allow
it, rather than five copies of the same data.

### 7.3 Build-time artifact pipeline

The `flutter_litert` pub package should primarily contain source, descriptors,
build tooling, notices, and hashes. It should not become a repository of every
large proprietary vendor SDK.

For every selected vendor, the build pipeline should:

1. resolve an exact LiteRT and vendor SDK version;
2. download from an official pinned URL or accept a user-supplied local SDK;
3. require explicit licence acceptance where necessary;
4. verify the archive SHA-256 before extracting;
5. extract an allowlist of required arm64 Android files only;
6. build missing open-source LiteRT compiler/dispatch plugins reproducibly;
7. verify output filenames, ELF architecture, dependencies, and hashes;
8. deduplicate common blobs;
9. emit either embedded APK assets or Play dynamic-feature modules;
10. produce a size and licence-notice report.

Build caching should avoid downloading multi-gigabyte SDK archives repeatedly,
but the cache must be keyed by version and hash.

### 7.4 Runtime device selection

Selection must use the SoC, not the phone brand. A Samsung-branded phone can
contain either Qualcomm Snapdragon or Samsung Exynos.

On API 31 or newer, use `Build.SOC_MANUFACTURER` and `Build.SOC_MODEL`, compare
them against exact normalized values in the pack descriptor, then probe any
required system driver library. A matching string alone is not sufficient
proof that the runtime can initialize.

Only one vendor pack and one SoC overlay may be selected for a process.

### 7.5 Pack availability abstraction

The runtime should depend on a small interface such as:

```text
NpuPackSource.ensureAvailable(selectedPack)
  EmbeddedPackSource -> locate, verify, and prepare signed APK blobs
  PlayFeatureSource  -> locate the installed split or request it on demand
```

After `ensureAvailable`, both sources return the same prepared-pack structure.
All later LiteRT code is delivery-neutral.

### 7.6 Safe preparation and caching

Embedded preparation should:

- use an app-private versioned directory;
- hold a process/file lock so Flutter isolates cannot race extraction;
- write into a temporary sibling directory;
- verify every SHA-256 and expected size;
- complete with an atomic rename;
- reject symlinks, path traversal, unexpected files, and wrong ELF ABIs;
- key the result by app version, LiteRT version, vendor SDK version, pack hash,
  SoC overlay, and relevant build fingerprint;
- retain only the selected pack when safe to do so.

The loader must account for Android linker namespaces and vendor libraries that
load dependencies by filename. This requires device tests; it is not merely a
ZIP extraction task.

### 7.7 Isolated compiler and dispatch discovery

LiteRT discovers compiler and dispatch plugins by scanning configured
directories. Several vendors in one flat directory are ambiguous and may cause
the wrong matching plugin to load first.

The environment passed to LiteRT must point
`CompilerPluginLibraryDir` and `DispatchLibraryDir` at the one selected pack's
isolated directory. No unrelated vendor plugin should be visible there.

### 7.8 Compiler cache

JIT compilation can be expensive. Cache compiled output by at least:

- source model hash;
- LiteRT version;
- vendor and SDK version;
- SoC model/runtime overlay;
- compilation options;
- device build fingerprint when required by the vendor.

A stale compiled artifact must never be reused merely because the model
filename stayed the same.

## 8. Dart API and downstream migration

Packaging a vendor runtime does not automatically move an existing
`Interpreter` call onto CompiledModel. Current consumers that construct the
classic Interpreter continue using its selected delegates, commonly CPU, until
they request or inherit the new session policy.

The target API should provide one process-wide initialization point and a
unified session factory:

```dart
final status = await FlutterLiteRt.initialize(
  acceleratorPreference: const [
    Accelerator.npu,
    Accelerator.gpu,
    Accelerator.cpu,
  ],
  npuModelPolicy: NpuModelPolicy.validatedOnly,
);

final session = await LiteRtSession.fromAsset(
  'assets/mobilefacenet.tflite',
);
```

Names are illustrative. Required behavior is:

- initialization is asynchronous because Play on-demand delivery or embedded
  preparation may be asynchronous;
- the chosen native path is process-wide and usable from worker isolates;
- strict NPU produces an actionable error when unavailable;
- NPU/GPU/CPU preference falls back only to accelerators the app allowed;
- the session exposes what actually happened.

Useful diagnostics include:

```text
requested accelerators
effective accelerators
detected SoC manufacturer/model
selected vendor and pack version
pack delivery source
compiler and dispatch loaded
fully accelerated
fallback reason
model validation status
```

Detection packages should migrate once to the shared session factory or make
their existing `useCompiledModel` default consult the process-wide policy. They
do not change again when MediaTek, Samsung, or another vendor is added.

Backward-compatible classic Interpreter entry points should remain available.
Do not silently reroute every old Interpreter construction to NPU because the
numerical behavior and supported operations differ.

## 9. Model-specific safety policy

Vendor readiness and model readiness must be separate records.

A central model policy can identify a model by SHA-256 and record:

```text
model hash
model semantic version
approved vendors/SoCs
reference backend
numeric tolerance
application-level acceptance result
known fallback requirement
date and test evidence
```

Recommended modes:

- `validatedOnly`: use NPU only when the exact model hash/vendor combination is
  approved;
- `preferNpu`: attempt NPU and expose diagnostics, intended for development;
- `strictNpu`: require NPU and fail otherwise, intended for hardware proof and
  controlled use;
- `disabled`: use the normal GPU/CPU route.

Production detection packages should default to `validatedOnly`, not blind
`preferNpu`.

## 10. Size findings and expectations

These figures are measurements or early engineering estimates, not final
release promises. Final size depends on exact SDK versions, ELF stripping,
compression, licence-required files, and whether Android must store a second
prepared copy.

| Vendor pack | Observed inputs | Preliminary compressed APK impact |
|---|---|---:|
| Qualcomm | Current generated runtime is about 105 MiB uncompressed per HTP generation; common files are repeated | Approximately 60 MiB for a deduplicated HTP-only multi-generation pack |
| MediaTek | Public SDK archive about 63.5 MiB compressed; v8+v9 Android adapter libraries recompressed to about 12.6 MiB | Approximately 13 MiB plus LiteRT compiler/dispatch plugins; may be lower if a compatible system adapter is sufficient |
| Samsung | SDK download about 38.5 MiB; relevant ARM libraries about 59 MiB uncompressed and 18.2 MiB recompressed | Approximately 18 MiB plus LiteRT plugins |
| Google Tensor | Compiler and dispatch plugins total about 1.1 MiB uncompressed | Small runtime, but every AOT model variant adds its own size |

Initial universal-APK estimates:

- Qualcomm plus MediaTek: roughly 70-80 MiB compressed;
- Qualcomm plus MediaTek plus Samsung: roughly 90-105 MiB compressed;
- Google Tensor: runtime overhead is small, but model-specific AOT artifacts
  can dominate.

These estimates must be replaced by a CI-generated size report before release.

Play AAB users normally download only the base app and their matching runtime,
so the total upload bundle can be large without charging every user for all of
it. Standalone universal APK users download every selected pack.

## 11. Licence and redistribution gates

Technical access to an SDK is not sufficient permission to redistribute it.

### Qualcomm

The current workflow downloads official QAIRT material during the build,
extracts an allowlist, and does not commit or republish the SDK. The final
application packaging rights and required notices still need to remain pinned
to the exact SDK licence/version used.

### MediaTek

The inspected NeuroPilot licence appears to permit object-code distribution
when incorporated into an application for MediaTek chipsets, while prohibiting
standalone redistribution of the SDK. This suggests that a signed final APK may
be a valid form but a raw MediaTek runtime ZIP on pub.dev or GitHub may not be.
This requires legal/licence confirmation before release; it is not legal
advice.

### Samsung

The downloaded SDK's licence material was not available as ordinary readable
text in the inspected package. Redistribution rights remain unresolved and are
a hard release gate.

### Google Tensor

The SDK is Beta and subject to its access and distribution terms. AOT model
artifacts also require a model-variant delivery policy.

The pack builder must support both official downloads and a user-supplied SDK
path. If automated redistribution or downloading is not permitted, the build
must stop with instructions rather than silently copying an unapproved binary.

## 12. Implementation roadmap

### Phase 0: preserve the Qualcomm checkpoint

- Keep current Qualcomm v73/v75/v79 runtime and physical tests green.
- Treat existing behavior as the baseline, not as the final multi-vendor
  packaging architecture.
- Record untested Qualcomm SoCs explicitly.

Exit condition: the current strict NPU and fallback tests remain reproducible.

### Phase 1: generic pack foundation

- Define and version the vendor-pack manifest.
- Implement content-addressed blobs and deterministic build reports.
- Implement SoC matching and required-system-library probes.
- Add `EmbeddedPackSource` and isolated preparation.
- Add safe locking, verification, cache invalidation, and diagnostics.
- Migrate Qualcomm into one common base plus generation overlays.
- Prove loading from the proposed app-private prepared path on physical
  Qualcomm devices. Use a signed-layout alternative if Android linker behavior
  requires it.

Exit condition: Qualcomm behaves exactly as before through the generic pack
interface, and the multi-generation APK does not expose competing plugins to
LiteRT.

### Phase 2: Play delivery adapter

- Generate Qualcomm dynamic-feature modules from the same manifests.
- Generate the SoC device-targeting configuration.
- Support conditional install-time delivery first.
- Build an optional on-demand `PlayFeatureSource` only after install-time works.
- Validate generated AABs with `bundletool` device specifications.
- Test an unsupported/default group and each supported Qualcomm group.
- Decide and implement the minimal settings-plugin or configure-command
  integration for Flutter apps.

Exit condition: an AAB installs only the expected Qualcomm module for each
device specification, while the standalone embedded APK continues to work.

### Phase 3: second vendor - MediaTek

MediaTek is the preferred second Android vendor because it expands coverage
substantially and supports JIT compilation of ordinary TFLite models.

- Resolve redistribution/licence handling.
- Build LiteRT's MediaTek compiler and dispatch plugins reproducibly.
- Determine whether supported phones provide the needed NeuroPilot adapter or
  whether an app-bundled adapter is required.
- Add MediaTek descriptors, SDK recipe, dependency probe, and notices.
- Package Qualcomm plus MediaTek in one embedded APK without a flat-directory
  collision.
- Generate a separate MediaTek Play feature.
- Validate strict NPU, fallback, lifecycle, and the model correctness matrix on
  physical MediaTek hardware.

Firebase's catalog showed `gts10pwifi` (Galaxy Tab S10+) as a possible physical
MediaTek target. The catalog does not expose the SoC directly, so the first
run must log and confirm `Build.SOC_MANUFACTURER` and `Build.SOC_MODEL` before
it counts as vendor evidence.

Exit condition: one standalone APK and one Play AAB both support validated
Qualcomm and MediaTek devices through the same Dart API.

### Phase 4: third vendor - Samsung Exynos

- Resolve Samsung SDK redistribution terms.
- Build and package LiteRT's Samsung compiler and dispatch plugins.
- Probe the device-provided `libenn_public_api_cpp.so` dependency.
- Add exact Exynos 2500/2600 SoC targeting.
- Find a confirmed physical Exynos target; a Samsung phone name alone is not
  evidence because regional models can contain Qualcomm SoCs.
- Run the complete strict, fallback, lifecycle, and correctness gates.

Exit condition: Samsung is another registry entry, not a special-case rewrite,
and both delivery modes can select it without changing detection packages.

### Phase 5: Google Tensor AOT

- Obtain and pin approved Tensor SDK tooling.
- Add Tensor G5 runtime targeting.
- Define a model-artifact registry mapping source model hashes to Tensor AOT
  variants.
- Use Play AI Packs for Play builds.
- Define signed embedded AOT-model packaging for standalone APK builds.
- Validate on a Pixel 10 family physical device; Firebase listed Pixel 10 Pro
  (`blazer`) as a candidate.

Exit condition: requesting an approved model selects its matching Tensor AOT
artifact; an uncompiled model falls back clearly rather than pretending to use
the NPU.

### Phase 6: future vendors

For every new LiteRT Android vendor:

1. add a descriptor and pack recipe;
2. pass licence review;
3. add embedded and Play packaging outputs;
4. add SoC and dependency probes;
5. validate on physical hardware;
6. approve models by hash;
7. publish size and readiness records.

No detector package should change merely because the registry gained a vendor.

Intel OpenVINO should use a separate desktop pack path rather than entering the
Android APK registry.

## 13. Verification strategy

### 13.1 Tests that do not consume Firebase quota

Run these before every physical test:

- schema and manifest unit tests;
- exact SoC selection and unsupported-device tests;
- duplicate/collision rejection;
- archive hash, allowlist, ABI, ELF dependency, and licence-notice audits;
- malicious path and corrupt extraction tests;
- concurrent initialization and Flutter isolate tests;
- cache invalidation tests;
- local emulator CPU/GPU fallback;
- AAB inspection and `bundletool` device-spec selection for every group;
- APK/AAB compressed and installed-size reports;
- deterministic pack rebuild/hash comparison.

### 13.2 Physical vendor gate

Each SoC generation needs:

1. logged SoC manufacturer/model and Android version;
2. strict `{npu}` construction that fails if NPU cannot take the graph;
3. LiteRT compiler, dispatch, and vendor execution evidence;
4. at least five repeated inferences;
5. at least three create/run/close cycles;
6. CPU-first initialization regression coverage;
7. mixed NPU plus CPU fallback behavior;
8. fixed-output comparison against bare CPU;
9. application-level validation for every production detector;
10. unsupported/missing-runtime error quality.

GPU fallback should also be tested where the requested policy includes GPU,
but a successful GPU fallback must never be reported as proof of NPU use.

### 13.3 Firebase Test Lab budget

The project has five free Test Lab runs per day. The Android NPU workflow must
remain manual and must never run on push or pull request.

Use the daily budget only after all local gates pass. A good allocation for a
new vendor day is:

1. one strict smoke run that also confirms the actual SoC;
2. one repeated lifecycle and fallback run after fixing any smoke issue;
3. one full representative model sweep;
4. one packaging-mode or second-device/generation check;
5. one reserved rerun for a real failure or final confirmation.

Do not spend all five on identical reruns. Preserve logs and artifacts from
every matrix so a failure can be diagnosed locally before consuming another
run. At the end of the Qualcomm testing day described by this document, all
five daily runs had been used; this research and documentation work consumed
no additional Test Lab runs.

## 14. Definition of vendor readiness

A vendor may be labelled **ready** only when:

- an exact supported-SoC list is documented;
- required runtime files can be acquired and built reproducibly;
- redistribution terms and notices are resolved;
- both intended delivery products pass binary and size audits;
- runtime selection isolates it from every other packaged vendor;
- at least one physical supported device proves strict NPU execution;
- each claimed SoC/runtime generation is either tested or clearly marked
  experimental/unvalidated;
- fallback works on unsupported devices and when runtime initialization fails;
- model compatibility is explicit and keyed to exact model artifacts;
- lifecycle, concurrency, cache, update, and corruption cases pass;
- effective backend and fallback reasons are observable to the app.

Under this definition, Qualcomm is validated for SM8550/v73, SM8650/v75, and
SM8750/v79, but the entire published Qualcomm SoC list is not yet fully ready.
MediaTek, Samsung, and Google Tensor are upstream-supported but are not yet
`flutter_litert`-ready.

## 15. Known risks and decisions still required

| Risk or decision | Current direction |
|---|---|
| App-private native library loading differs across Android/vendor versions | Prove it in Phase 1; retain a signed native-layout fallback |
| Standalone APK size | Deduplicate common blobs, let apps choose vendors, publish measured reports |
| Several compiler/dispatch plugins conflict | Expose only one isolated selected directory to LiteRT |
| Play device targeting is Beta | Validate with `bundletool`, keep embedded mode independent, test real Play delivery before declaring production-ready |
| Dynamic features require app-level Gradle integration | Provide a settings plugin or generator; no copied vendor logic |
| Proprietary SDK redistribution | Licence gate per exact version; support local SDK input |
| Runtime succeeds but model output changes | Exact-model CPU golden and application quality gates |
| Existing plugins use classic Interpreter | One migration to a shared session/global policy; preserve backward compatibility |
| Google Tensor has no public JIT path | Add only with an AOT artifact registry, or revisit when upstream JIT arrives |
| Test Lab does not expose SoC in its catalog | Confirm the SoC inside the first physical run |
| Test Lab may install the default AAB group | Verify targeting locally with `bundletool`; derive an exact-module test APK when needed and do not mislabel it as Play-serving proof |

## 16. Final implementation direction

The intended end state is:

- `flutter_litert` owns every vendor pack and all selection logic;
- Agelapse releases one signed, offline, multi-vendor standalone APK;
- Play consumers release one AAB whose users receive only the correct SoC
  runtime through Play Feature Delivery;
- AOT models use the same registry and Play AI Packs where applicable;
- detector packages select a shared accelerator policy, not native files;
- adding vendor three, four, or ten means adding one pack implementation and
  its evidence, without redesigning the system.

The next engineering work should begin with Phase 1: define the pack schema,
migrate the tested Qualcomm implementation into it, and prove safe isolated
loading from a multi-generation standalone APK. MediaTek should be the second
vendor once its build and licence gates are resolved.
