# flutter_litert 3.9.0 release and dependent-package plan

Status: draft for review. Do not execute until Hugo approves this plan.

Last reviewed: 2026-09-19

## Non-negotiable boundaries

- Codex may prepare, validate, and commit release-ready changes on the feature
  branch after this plan is approved.
- Codex must **not** run `dart pub publish`, publish a GitHub release, create or
  push a release tag, or otherwise publish `flutter_litert 3.9.0`.
- Step 3 is a hard manual-publication gate. Stop after the final readiness
  report and wait for Hugo to publish `flutter_litert 3.9.0` himself.
- Do **not** modify, release, or publish any dependent detection package in
  Step 4 or later until Hugo explicitly confirms that `flutter_litert 3.9.0`
  is live on pub.dev.
- Every dependent-package publication is also manual. Codex can prepare each
  package to the same approval point, but Hugo performs the actual publish.
- Keep changes on feature branches and in small, reviewable commits. Approval
  of this plan authorizes pushing the release feature branch and opening or
  updating a draft PR solely to run the required CI. It does not authorize a
  merge, squash, release tag, publication, or history rewrite.
- Do not close issues #17, #18, or #19 or post the final replies until the fix
  is published and Hugo approves the exact issue replies. Preparing draft
  replies is allowed before publication.

## Release definition of done

`flutter_litert 3.9.0` is ready for Hugo's manual publish only when all of the
following are true:

- Issues #17, #18, and #19 meet their acceptance criteria below.
- The exact Xcode 27 / iOS 27 simulator build completes successfully, including
  the final link step. Merely getting past the reported CMake error is not
  enough.
- All current package, podspec, example, documentation, and API references use
  `3.9.0` where they describe the pending release. Historical changelog entries
  and historical benchmark records retain the versions they actually measured.
- Dependency resolution is reproducible in every checked-in package or test
  host, with no unintended path, Git, or dependency override in the package
  that pub.dev will publish.
- Formatting, static analysis, unit tests, required integration tests, platform
  builds, documentation generation, package scoring, and CI are clean.
- `dart pub publish --dry-run` exits successfully with zero errors and zero
  warnings, and its package-content listing has been manually reviewed.
- The branch is clean, the final commits are reviewable, and the readiness
  report records commands, versions, results, skipped tests, and any remaining
  limitations.

## Known baseline to verify, not assume

- Working branch: `feature/issues-17-19-review`.
- Intended package version: `3.9.0`.
- Intended toolchain: Flutter 3.47.5, Dart 3.13.4, macOS 26.7, Xcode 27.0
  build 27A266a, and the iOS 27 simulator runtime.
- The branch currently contains separate commits for the #17 documentation,
  #18 policy API, #19 diagnosis/configuration, 3.9.0 metadata, Flutter analyzer
  migrations, and refreshed nested-example dependencies.
- The last observed local checks reported all three analyzers clean and 293
  native tests passing with 3 skips. These results must be rerun after the
  final edits; they are not final release evidence.
- Issues #17, #18, and #19 are still open. Issue #19 currently has only the
  acknowledgement comment.

## Step 0: establish a clean, reproducible release workspace

1. Confirm the branch and repository state:

   ```sh
   git status --short --branch
   git log --oneline --decorate -15
   git diff --check main...HEAD
   ```

2. Record the exact environment in the readiness report:

   ```sh
   sw_vers
   flutter --version
   dart --version
   xcodebuild -version
   xcode-select -p
   xcrun simctl list runtimes
   pod --version
   cmake --version
   ```

3. Require `flutter doctor -v` to recognize Xcode 27 and the installed iOS 27
   simulator. Investigate every warning relevant to a supported platform. The
   known duplicate-ADB warning is unrelated to this release but should be
   recorded rather than silently omitted.

4. Fetch remote Git state and compare the branch base with `origin/main`.
   Incorporate any new main-branch changes deliberately. Do not hide conflicts
   or overwrite local work.

5. Resolve dependencies from each independent package root so analysis never
   relies on stale `.dart_tool/package_config.json` files:

   ```sh
   flutter pub get
   (cd example && flutter pub get)
   (cd example_web && flutter pub get)
   (cd example/flex_test_host && flutter pub get)
   ```

6. Review every resulting tracked lockfile or analyzer migration. Commit only
   intentional changes, separated from implementation fixes.

## Step 1: finish and verify issues #17, #18, and #19

### 1A. Issue #17 — native `InterpreterFactory` documentation

Acceptance criteria:

- Every `InterpreterFactory` and `InterpreterPool` example that requires native
  types imports `package:flutter_litert/native.dart`.
- The primary `package:flutter_litert/flutter_litert.dart` entry point remains
  portable and does not expose native-only types on web.
- Documentation states the supported native platforms and points portable code
  to `CompiledModel.fromBufferWithConfigAsync` with
  `CompiledModelConfig.auto()` where appropriate.
- README anchors, API comments, and snippets agree; no stale snippet tells web
  code to use a native-only class.
- A small temporary analyzer probe or existing compile test proves the
  documented native imports resolve and the portable import remains valid on
  web. Do not publish the temporary probe.

Checks:

```sh
rg -n "InterpreterFactory|InterpreterPool" README.md lib test example example_web
flutter analyze .
```

Prepare a concise issue reply explaining the corrected import and portable
alternative. Do not post or close the issue before Step 3 and Hugo's approval.

### 1B. Issue #18 — `CompiledModel` automatic policy

Review the complete public API rather than only the happy path:

- `CompiledModelConfig` and `CompiledModelPolicy` are exported from the correct
  portable library surface.
- Policies cover `auto`, `cpu`, strict `gpu`, `gpuWithCpuFallback`, strict
  `npu`, and `npuWithCpuFallback` on native, web, and unsupported-platform
  implementations with matching signatures.
- Existing constructors keep their documented behavior and remain source
  compatible.
- `auto` has one precise contract: strict fp32 GPU first, then a complete
  CPU-only retry if construction fails. It does not silently become a mixed
  graph, benchmark accelerators, select NPU, or imply numerical correctness.
- Strict policies do not silently retry on CPU. Mixed policies and complete
  retry behavior are distinguishable.
- `requestedConfig`, `requestedAccelerators`, `accelerators`, `didFallback`,
  and `isFullyAccelerated` have consistent, immutable, documented semantics.
- Web covers WebGPU, WASM fallback, watchdog behavior, unsupported strict NPU,
  and NPU-with-CPU fallback.
- Native tests cover CPU, successful GPU, failed GPU to CPU, mixed GPU/CPU,
  successful and failed NPU cases where the platform permits them.
- Unsupported-platform stubs remain API-compatible and fail clearly.
- README examples and the 3.9.0 changelog describe limitations and migration
  choices without overstating acceleration or correctness.

Required focused tests, followed later by the full suite:

```sh
flutter test test/compiled_model_config_test.dart
flutter test test/compiled_model_smoke_test.dart
flutter test test/compiled_model_npu_macos_test.dart
flutter test test/compiled_model_unsupported_test.dart
flutter test --platform chrome test/compiled_model_web_test.dart
```

Prepare a concise issue reply with the new constructors, policy table, and
fallback diagnostics. Do not post or close the issue before Step 3 and Hugo's
approval.

### 1C. Issue #19 — Xcode 27 CMake deployment-target failure

Use the exact installed environment. Do not treat a build under Xcode 26 as
final evidence.

1. Confirm Xcode 27, the iOS 27 SDK, and the simulator runtime:

   ```sh
   xcodebuild -version
   xcodebuild -showsdks
   xcrun simctl list runtimes
   flutter doctor -v
   ```

2. Reconfirm the diagnosis:

   - The reported iOS 12 value came from DartCV's CMake hook, not
     `.github/workflows/build-coreml-macos.yml`.
   - `opencv_dart` resolves to at least 2.2.2.
   - `dartcv4` resolves to at least 2.3.1, which contains the configurable
     deployment-target fix.
   - The example is a root package and therefore may set
     `hooks.user_defines.dartcv4.ios.deployment_target: '15.0'`.
   - Do not claim that this `user_defines` setting propagates from a dependency
     to consuming apps. Dart only honors it from the root package or workspace.

3. Verify all relevant iOS settings:

   - Example Runner target: 15.0.
   - DartCV hook target: 15.0.
   - `ios/flutter_litert.podspec`: retain 13.0 only if the complete Xcode 27
     build proves it is supported independently of DartCV. If Xcode 27 rejects
     it, raise the package minimum consistently and document the compatibility
     change before release.
   - No generated or checked-in setting reintroduces 12.0.

4. Remove stale build products that could mask the result, resolve the example,
   and run a verbose simulator build:

   ```sh
   (cd example && flutter clean)
   (cd example && flutter pub get)
   (cd example && flutter build ios --simulator --no-codesign -v)
   ```

5. Preserve the relevant log as release evidence and prove:

   - The DartCV/OpenCV CMake invocation uses `-DDEPLOYMENT_TARGET=15.0`.
   - No `CMAKE_TRY_COMPILE` target uses iOS 12.
   - The native-asset build succeeds for every simulator architecture the
     toolchain actually requests.
   - The final Runner link succeeds.

6. If `_tflite_plugin_create_delegate`, `_tflite_plugin_destroy_delegate`, or
   any other linker failure remains, diagnose and fix it in a separate commit.
   Rerun the clean build afterward. A CMake success followed by a linker failure
   does not pass this gate.

7. Run the relevant iOS integration tests on the iOS 27 simulator, matching CI
   where practical:

   ```sh
   (cd example && flutter test integration_test/compiled_model_test.dart -d <IOS_27_UDID> --timeout 600s)
   (cd example && flutter test integration_test/ios_compiled_model_npu_test.dart -d <IOS_27_UDID> --timeout 600s)
   (cd example && flutter test integration_test/interpreter_delegate_fallback_test.dart -d <IOS_27_UDID> --timeout 600s)
   (cd example && flutter test integration_test/interpreter_coexistence_test.dart -d <IOS_27_UDID> --timeout 600s)
   ```

   Record expected simulator limitations, especially that a simulator does not
   validate physical-device Neural Engine execution.

8. Prepare an issue reply that identifies DartCV as the source, lists the fixed
   versions and root hook configuration, and reports the exact Xcode 27 build
   result. Do not post or close the issue before Step 3 and Hugo's approval.

### Step 1 gate

Stop and fix any failure. Do not call the issues complete merely because code
exists. The implementation, docs, focused tests, clean Xcode 27 build, and
draft issue replies must all be reviewable.

## Step 2: complete every `flutter_litert 3.9.0` pre-publish check

### 2A. Version and release-content audit

1. Verify read-only on pub.dev that `3.9.0` has not already been published and
   that the intended version is available. Pub.dev versions are immutable.

2. Verify `3.9.0` in every current release location:

   - `pubspec.yaml`
   - `ios/flutter_litert.podspec`
   - `macos/flutter_litert.podspec`
   - consumer-facing README dependency snippets
   - `example/pubspec.yaml`
   - `example_web/pubspec.yaml`
   - `example/flex_test_host/pubspec.yaml`
   - API documentation examples such as the Flex add-on snippet
   - the first `CHANGELOG.md` heading and its complete #17/#18/#19 notes

3. Search the whole tracked tree for stale current-release references:

   ```sh
   git grep -n -E "3[.]8[.][0-9+]*|\^3[.]8|version 3[.]8|v3[.]8"
   git grep -n "3.9.0"
   ```

4. Classify every match. Update stale installation instructions, examples,
   generated package metadata, and current release prose. Preserve accurate
   historical changelog headings, migration notes, tags, and benchmark records;
   rewriting history to say it measured 3.9.0 would be incorrect.

5. Confirm the 3.9.0 changelog includes:

   - corrected native utility imports (#17);
   - the new policy API and exact fallback semantics (#18);
   - the DartCV/OpenCV CMake diagnosis and Xcode 27 verification (#19);
   - minimum Flutter/Dart/iOS changes, if any;
   - dependency changes and user action required for root DartCV hook settings;
   - compatibility and migration notes for existing constructors.

6. Review README snippets as code: imports, async/sync APIs, platform scope,
   version constraints, links, headings, and terminology must match the shipped
   API.

### 2B. Dependency and SDK audit

1. Run and review, without blindly upgrading unrelated major versions:

   ```sh
   flutter pub get
   flutter pub outdated
   dart pub deps --style=compact
   ```

   Treat any security advisory reported during dependency resolution as a
   blocking review item. Upgrade the affected dependency or document why the
   advisory cannot affect this package; do not silently add an
   `ignored_advisories` entry.

2. Repeat dependency resolution in `example`, `example_web`, and
   `example/flex_test_host`.

3. Verify:

   - hosted dependencies use intentional lower and upper bounds;
   - the published root package has no path, Git, or dependency override;
   - example-only path overrides cannot leak into the published dependency
     metadata;
   - Flutter, Dart, CocoaPods, Gradle, CMake, Java, and platform minimums agree
     with the code and documented support matrix;
   - `meta 1.19.0`, matcher/test packages, OpenCV, and DartCV resolve as expected
     in the test hosts that use them;
   - lockfile changes are intentional and deterministic.

4. Test the lowest allowed dependency resolution where Flutter tooling supports
   it. Restore the normal resolution afterward and rerun the main checks:

   ```sh
   flutter pub downgrade
   flutter analyze .
   flutter test
   flutter pub get
   ```

### 2C. Formatting and static analysis

Run formatting before the final test pass so generated formatting changes are
included in what is tested:

```sh
dart format --output=none --set-exit-if-changed .
```

If it reports changes, run `dart format .`, review every diff, commit the
formatting with the relevant implementation or as a separate mechanical commit,
and rerun the no-change command. The final command must exit zero.

Run analysis from every package context:

```sh
dart analyze
flutter analyze .
(cd example && flutter analyze)
(cd example_web && flutter analyze)
(cd example/flex_test_host && flutter analyze)
```

Both Dart and Flutter analysis must report zero issues. Do not suppress a new
diagnostic merely to make the release pass unless the suppression is narrow,
justified, and reviewed.

### 2D. Tests and supported-platform builds

Required local checks on this Mac:

```sh
flutter test
flutter test --platform chrome \
  test/compiled_model_web_test.dart \
  test/web_detector_utils_web_test.dart \
  test/litert_web_loader_test.dart \
  test/util/camera_frame_prepare_test.dart \
  test/util/camera_frame_test.dart \
  test/util/yuv_conversion_test.dart \
  test/util/cover_fit_transform_test.dart
(cd example && flutter build macos)
(cd example && flutter build ios --simulator --no-codesign)
```

Also run the macOS integration suites used by CI, including CompiledModel,
Interpreter coexistence, Metal, XNNPACK, Core ML, and the Flex host. Record any
hardware-dependent skip explicitly.

Run Android build/tests locally if the installed SDK and remaining disk space
permit; otherwise require the equivalent CI jobs to pass. Linux and Windows
behavior must be covered by CI. A release is not ready while required CI jobs
are pending, skipped unexpectedly, or red.

Require the full GitHub Actions workflow to pass on the final commit, including:

- macOS, Linux, Windows, Android, iOS, and web jobs;
- published iOS-framework verification;
- format and analyzer checks;
- native library packaging checks;
- browser and platform integration tests.

Review whether the iOS CI job should add an Xcode 27 lane. Its existing Xcode
16.4 coverage is useful compatibility evidence but cannot substitute for the
local Xcode 27 reproduction required by #19.

### 2E. Native package and artifact validation

1. Validate the consumer artifacts rather than only working-copy binaries:

   ```sh
   ./scripts/verify_published_ios_frameworks.sh
   ```

2. Validate the podspecs with CocoaPods. Investigate all failures and warnings;
   do not hide meaningful warnings behind `--allow-warnings`:

   ```sh
   pod lib lint ios/flutter_litert.podspec
   pod lib lint macos/flutter_litert.podspec
   ```

   If CocoaPods requires a documented Flutter-plugin-specific lint flag, record
   the exact reason and command in the readiness report.

3. Parse both Swift packages and verify their binary-target URLs/checksums and
   platform declarations:

   ```sh
   (cd ios/flutter_litert && swift package dump-package)
   (cd macos/flutter_litert && swift package dump-package)
   ```

4. Verify expected architectures, exported symbols, checksums, download URLs,
   licenses, and package resources. Confirm the pub archive excludes the large
   iOS XCFrameworks and that the podspec download path remains reproducible.

5. Build from a clean consumer-style checkout or temporary package copy, not
   only the repository with cached frameworks. This must exercise the same
   archive contents and download paths a pub.dev consumer receives.

### 2F. Documentation, package contents, and pub.dev quality

1. Generate API documentation and treat unresolved references or generation
   errors as failures:

   ```sh
   dart doc
   ```

2. Verify package essentials: `LICENSE`, `README.md`, `CHANGELOG.md`, AUTHORS,
   repository/homepage/issue URLs, description, topics, platform declarations,
   public API exports, and example source.

3. Review `.pubignore` and the exact archive listing. It must exclude build
   products, caches, credentials, local plans, large test/model assets, local
   overrides, and unpublished binaries while retaining every runtime file,
   license, source, and example needed by consumers. Add this plan to
   `.pubignore` or remove it from the release tree before the final dry run.

4. Run a local pub.dev quality analysis with the current `pana` release and
   review every lost point or warning. Record the exact tool version and result.

5. Check tracked files for accidental secrets, absolute developer-machine
   paths, debug logging, temporary probes, generated archives, and symlinks that
   will not survive publication.

### 2G. Final pub.dev dry run — blocking gate

From the repository root, on the exact final commit:

```sh
dart pub publish --dry-run
```

Requirements:

- Exit code 0.
- Zero errors.
- Zero warnings.
- Package name and version are exactly `flutter_litert 3.9.0`.
- The archive content list and compressed/uncompressed size are plausible.
- No path dependency, override, cache, secret, build output, local plan,
  excluded XCFramework, or unrelated file is included.
- README, changelog, license, example, Dart sources, native build files, and
  required runtime resources are included.

If anything changes after this command, rerun formatting, analysis, affected
tests/builds, package-quality checks, and the dry run. A stale dry run does not
count.

### 2H. Final Git and review gate

1. Review every commit and the complete `main...HEAD` diff.
2. Ensure each implementation, documentation, dependency, and mechanical
   migration is in an understandable commit.
3. Run `git diff --check` and require a clean working tree.
4. Confirm the final commit is the one tested locally and by CI.
5. Prepare, but do not publish:

   - release notes based on `CHANGELOG.md`;
   - exact draft replies for issues #17, #18, and #19;
   - the proposed PR title/body if a PR is desired;
   - the exact manual publish command for Hugo.

## Step 3: manual `flutter_litert 3.9.0` publication gate

Codex stops and reports back with:

- final commit SHA and clean-tree status;
- version/toolchain matrix;
- every command run and its result;
- test totals and explicit skips;
- Xcode 27 CMake and final-link evidence for #19;
- CI run links/statuses;
- `pana` result;
- full `dart pub publish --dry-run` result and package-size summary;
- remaining risks or limitations;
- draft issue replies and release notes.

Hugo reviews the report and manually runs the real publication command. Codex
must not run it. Use the normal interactive validation and confirmation flow;
do not use `--force` or `--skip-validation` to bypass the final checks.

After Hugo says publication succeeded, perform read-only verification that
pub.dev serves version 3.9.0, its documentation renders, dependency metadata is
correct, and the archive is installable in a fresh consumer app. If Hugo then
approves the prepared GitHub communication, post the issue replies and close
#17, #18, and #19. Create/push tags or a GitHub release only if Hugo separately
authorizes those actions.

**Hard stop:** do not begin Step 4 until Hugo explicitly confirms that
`flutter_litert 3.9.0` is live on pub.dev.

## Step 4: common audit for all seven dependent detection packages

Packages and current local versions at the time this plan was drafted:

| Package | Current version | Current `flutter_litert` | Relationship |
|---|---:|---:|---|
| `animal_detection` | 4.0.0 | `^3.8.0` | Must be released before cat and dog |
| `cat_detection` | 4.0.0 | `^3.8.0` | Depends on `animal_detection ^4.0.0` |
| `dog_detection` | 4.0.0 | `^3.8.0` | Depends on `animal_detection ^4.0.0` |
| `face_detection_tflite` | 6.8.0 | `^3.8.0` | Independent after flutter_litert |
| `pose_detection` | 3.7.0 | `^3.8.0` | Independent after flutter_litert |
| `hand_detection` | 4.1.0 | `^3.8.0` | Independent after flutter_litert |
| `object_detection` | 1.0.0 | `^3.8.0` | Independent after flutter_litert |

For each repository, create a feature branch and first audit the real source,
pubspecs, examples, CI, platform files, changelog, README, and public API. Then:

1. Select the next package version deliberately. Raising Flutter, Dart, or iOS
   minimums affects compatibility; do not assume a patch version without
   reviewing that package's versioning policy.
2. Set the hosted dependency to `flutter_litert: ^3.9.0` everywhere current
   release metadata or consumer examples require it. Remove obsolete local
   overrides before dry-run validation.
3. Upgrade `opencv_dart` from `^2.2.1+4` to the verified compatible release
   (`^2.2.2` at plan time).
4. Ensure resolution cannot fall below the DartCV CMake fix. If
   `opencv_dart`'s constraint does not guarantee it, add the intentional direct
   compatibility dependency `dartcv4: ^2.3.1` and document why it is direct
   even if no Dart source imports it.
5. Configure
   `hooks.user_defines.dartcv4.ios.deployment_target: '15.0'` in each root app,
   package workspace, or example that performs the native build. Do not claim a
   published dependency can impose this default on its consumers: official Dart
   hook behavior allows only the root package/workspace to supply user-defines.
   Put the required consumer configuration in the README and changelog.
6. Raise the minimum Flutter version to at least 3.47.5 wherever DartCV 2.3.1
   and Flutter's test-package `meta` pin must coexist.
7. Align the Dart minimum with the actual language/API requirements and with
   `flutter_litert 3.9.0` (at least Dart 3.10 unless a stricter package-specific
   minimum is required).
8. Audit `meta` correctly:

   - `pose_detection` and `hand_detection` directly import `package:meta`; keep
     it as a direct dependency and raise the lower bound to `^1.19.0` after
     solver validation.
   - Face, animal, cat, dog, and object currently declare `meta: ^1.17.0` but no
     `package:meta` import was found in `lib`, `test`, or `example`. Confirm with
     a fresh search, then remove the unused direct dependency rather than
     mechanically bumping it.
   - Remember that `^1.17.0` already permits 1.19.0. The real incompatibility
     was the older Flutter SDK/test-package pin, so the Flutter floor matters.

9. Align iOS project targets, CocoaPods metadata, DartCV hook configuration,
   documentation, and example apps with the verified iOS 15 build requirement.
10. Refresh tracked lockfiles intentionally. Review all transitive changes,
    especially `meta`, `matcher`, `test_api`, `vector_math`, `opencv_dart`, and
    `dartcv4`.
11. Update changelog, README installation snippets, CI Flutter version, example
    dependencies, compatibility notes, and every current version reference.
12. Run the same quality gate in each repository: formatting, all analyzers,
    unit/integration tests, web tests where supported, iOS 27 simulator build,
    other supported-platform builds/CI, `dart doc`, `pana`, and a zero-warning
    `dart pub publish --dry-run`.
13. Keep dependency/toolchain changes, source migrations, docs, and mechanical
    lockfile/analyzer changes in separate commits where that improves review.
14. Stop at a readiness report. Hugo publishes each package manually.

## Step 5: publish-ready `animal_detection`, then stop for manual publication

After `flutter_litert 3.9.0` is confirmed live:

1. Apply the Step 4 audit to `animal_detection` first.
2. Validate against the hosted `flutter_litert 3.9.0`, not only a path checkout.
3. Reach a clean, zero-warning dry run and report the proposed animal version.
4. Stop. Hugo manually publishes the new `animal_detection` release.
5. Verify the new animal version is live and resolvable from pub.dev before
   preparing cat or dog for final publication.

## Step 6: update `cat_detection` and `dog_detection`

Only after Hugo confirms the new animal release is live:

1. Update both packages to the newly published `animal_detection` constraint.
2. Also pin `flutter_litert: ^3.9.0` directly because both packages import it
   directly; do not rely only on the transitive animal dependency.
3. Apply every common Step 4 dependency, SDK, iOS, documentation, CI, and
   validation check independently to cat and dog.
4. Confirm their public re-exports of animal types remain compatible.
5. Produce separate readiness reports and stop for Hugo to publish each package
   manually.

## Step 7: update face, pose, hand, and object detection

After `flutter_litert 3.9.0` is live, these packages can be prepared
independently. Apply Step 4 fully to each:

- `face_detection_tflite`
- `pose_detection`
- `hand_detection`
- `object_detection`

Pay particular attention to:

- face's native and web examples and its existing Dart >=3.10 floor;
- pose's direct `package:meta` imports;
- hand's direct `package:meta` imports and older Dart >=3.6 floor;
- object's direct use and re-export of `flutter_litert` types;
- all packages' OpenCV/DartCV iOS native-asset builds under Xcode 27.

Each repository ends at a clean, zero-warning dry run and a readiness report.
Hugo performs every real pub.dev publication.

## Step 8: ecosystem verification after all manual publications

After Hugo confirms all seven releases are live:

1. Create a fresh consumer application with no path overrides or cached local
   packages.
2. Resolve each detection package from pub.dev and inspect `dart pub deps`.
3. Confirm the expected `flutter_litert`, OpenCV, DartCV, and `meta` versions.
4. Build representative native and web examples, including an Xcode 27 / iOS
   27 simulator build using the documented root hook configuration.
5. Verify cat and dog resolve the newly published animal release.
6. Check every pub.dev page, generated API docs, changelog, score, platforms,
   and repository links.
7. Record the final compatibility matrix and any follow-up work without
   rewriting the evidence from the release checks.

## Official references used by this plan

- [Dart package publishing](https://dart.dev/tools/pub/publishing)
- [`dart pub publish` and `--dry-run`](https://dart.dev/tools/pub/cmd/pub-lish)
- [Dart hook configuration and root-only user-defines](https://dart.dev/tools/hooks#configure-user-defines)
- [Dart/pub security advisories](https://dart.dev/tools/pub/security-advisories)
