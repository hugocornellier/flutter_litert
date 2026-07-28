# ModelStore: shared on-demand model delivery for flutter_litert plugins

Status: DRAFT (design review, nothing implemented)
Date: 2026-07-13
Owner: hugocornellier

## 1. Problem

Detection plugins built on flutter_litert (face_detection_tflite, hand_detection,
pose_detection, cat_detection, dog_detection, animal_detection, object_detection)
declare their .tflite models as Flutter assets. Flutter bundles every asset of
every package in the dependency graph into every build, on every platform, with
no tree shaking.

Measured impact on AgeLapse (imports face, hand, pose, cat, dog, and animal
transitively):

| Package              | Models on disk | Used by AgeLapse                          |
|----------------------|---------------:|-------------------------------------------|
| cat_detection        | 71 MB          | Only for cat projects                      |
| dog_detection        | 71 MB          | Only for dog projects                      |
| animal_detection     | 22 MB          | Via cat/dog full mode only                 |
| pose_detection       | 47 MB (38 kept)| Only for pose projects (heavy + yolov8n)   |
| face_detection_tflite| 29 MB (11 kept)| Core flow (back, landmark, iris, facenet)  |
| hand_detection       | 8 MB (0 kept)  | Never (only a version string is read)      |

AgeLapse currently fights this with five hand-maintained strip hacks (Android
Gradle doLast, iOS and macOS Xcode shell phases, Windows and Linux CMake exclude
lists), all citing a strip-savings.md audit that no longer exists in the repo.
Even after stripping, ~213 MB of models ship to every user, of which a typical
face-project user exercises ~11 MB.

Separately, face_detection_tflite PR #12 adds a `loadModelBytes` callback plus a
`ReleaseModelLoader` (GitHub release download, cache, SHA-256 table). It is the
right seam but the wrong home: the downloader would be private to one package,
and five sibling packages would need to copy it.

## 2. Goals

1. One shared, tested utility in flutter_litert that any downstream plugin with
   model files can adopt: download, cache, verify, dedupe, progress, mirrors.
2. Binary size drops on every platform for models the app author chooses not to
   bundle, with no per-app build hacks.
3. Bundled-asset behavior remains available and remains the offline path.
4. Works on Android, iOS, macOS, Windows, Linux, and web.
5. Serves PR #12's use case (strip face models, download on demand) with the
   downloader logic living in exactly one place.

Non-goals: Play Asset Delivery / iOS On-Demand Resources integration, model
encryption or DRM, delta updates, telemetry.

## 3. Verified facts the design rests on

All verified 2026-07-13 unless noted.

1. **GitHub release assets are NOT usable from browsers cross-origin.** The 302
   from `github.com/<o>/<r>/releases/download/...` carries no
   `Access-Control-Allow-Origin`, and neither does the final
   `release-assets.githubusercontent.com` response. Browser `fetch()` fails the
   CORS check. Native (dart:io) is unaffected. PR #12's web path would fail in
   a real cross-origin deployment.
2. **`raw.githubusercontent.com` sends `access-control-allow-origin: *`**, so
   files committed to the repo (all model files are committed, no LFS) are
   fetchable from browsers when pinned to a tag or commit.
3. **Release asset host supports `accept-ranges: bytes`**, so resume via Range
   requests is possible.
4. **Flutter supports per-platform asset bundling in package pubspecs.**
   `platforms:` on an assets entry (flutter/flutter#176393) filters package
   assets by target platform; verified in flutter_tools source that dependency
   packages go through the same `matchesPlatform` filter as app assets. First
   stable release: **Flutter 3.41 (Nov 2025)**. Using it in a package pubspec
   raises that package's minimum Flutter version. Bonus: `flutter test` runs
   as `TargetPlatform.tester`, which matches every platform filter, so
   platform-restricted assets remain visible to widget/unit tests.
5. flutter_litert currently depends only on path, quiver, ffi, web (plus SDK
   packages). It is an ffiPlugin on all native platforms and already has a web
   plugin.

## 4. Design overview

Two halves. Neither works alone.

**Half A (runtime, lives in flutter_litert):** a `ModelStore` library, exposed
as `package:flutter_litert/model_store.dart` (separate import; the core
interpreter API does not re-export it). Downloads model files described by
`ModelSpec` entries, verifies SHA-256, caches on disk, dedupes, reports
progress, fails over across mirror URLs.

**Half B (bundling, lives in each detection package):** model asset
declarations change from unconditional to `platforms: [web]`. Native platforms
stop bundling models entirely; web keeps them as bundled assets (Flutter web
fetches assets lazily over HTTP from the app's own origin, so unused models
cost hosting space, not user download time, and there is no CORS exposure).
Native code paths resolve models asset-first with download fallback, so the
same package version works bundled (web, or any app that re-declares the files
as its own assets) and unbundled (native default).

### Why asset-first resolution is the keystone

Every load goes through one resolver:

1. Check `AssetManifest` (or try `rootBundle.load`) for the bundled asset path.
   Present: return it. This covers web builds, apps that opted back into
   bundling, and tests.
2. Absent: `ModelStore.fetch(spec)` downloads (or serves from cache) and
   returns the bytes/path.

Consequences: no configuration is required for the common cases, the offline
path survives, an app can re-bundle any subset by declaring the files as its
own assets, and PR #12-style `loadModelBytes` remains a clean override hook in
front of both steps.

## 5. API sketch

```dart
// package:flutter_litert/model_store.dart

/// Compatible with face_detection_tflite PR #12.
typedef ModelBytesLoader = Future<Uint8List> Function(String fileName);

class ModelSpec {
  final String fileName;      // e.g. 'cat_face_landmarks_full.tflite'
  final List<Uri> urls;       // ordered: primary, then backups/mirrors
  final String sha256;        // required, lowercase hex
  final int sizeBytes;        // required, exact
  const ModelSpec({required this.fileName, required this.urls,
                   required this.sha256, required this.sizeBytes});
}

class ModelStore {
  /// Global default; packages use this unless handed another instance.
  static ModelStore instance = ModelStore();

  ModelStore({
    Directory? cacheDir,          // default: per-platform, see section 6
    HttpClient Function()? httpClientFactory,
    Uri Function(ModelSpec spec, Uri url)? rewriteUrl, // global mirror/base override
    RetryPolicy retryPolicy = const RetryPolicy(),     // attempts, backoff, timeouts
  });

  Future<CachedModel> fetch(ModelSpec spec,
      {void Function(int received, int total)? onProgress,
       CancellationToken? cancel});

  Future<void> prefetch(Iterable<ModelSpec> specs,
      {void Function(int received, int total)? onProgress,
       CancellationToken? cancel});

  Future<bool> isCached(ModelSpec spec);
  Future<void> seedFromDirectory(Directory dir, Iterable<ModelSpec> specs); // CI/tests
  Future<void> evict(ModelSpec spec);

  /// GC after version bumps. App-only API: the caller must pass the union of
  /// every package's manifest. A package must never call this on the shared
  /// instance or it deletes sibling packages' cached models.
  Future<void> evictExcept(Iterable<ModelSpec> keep);
}

class CachedModel {
  final ModelSpec spec;
  final String? filePath;          // null on web
  Future<Uint8List> readBytes();
}

/// The one entry point detection packages call.
class ModelResolver {
  static Future<Uint8List> resolve({
    required String assetPath,     // 'packages/<pkg>/assets/models/<file>'
    required ModelSpec spec,
    ModelBytesLoader? loadModelBytes, // app override, tried first if provided
    ModelStore? store,                // default ModelStore.instance
  });
}
```

Typed failures so apps can build UX: `ModelNetworkException` (per-URL attempt
log attached), `ModelChecksumException`, `ModelStorageException` (disk full,
quota), `ModelCancelledException`.

### Backup URLs / mirrors

Three layers, all trustless because SHA-256 verification is mandatory and the
hash is pinned in the package, not fetched:

1. `ModelSpec.urls` is an ordered list. The store walks it: for each URL,
   attempt with retry/backoff per `RetryPolicy`; on exhaustion move to the next.
2. `rewriteUrl` on the store lets an app globally redirect to a corporate
   mirror or self-hosted bucket without touching package manifests.
3. `loadModelBytes` bypasses the store entirely (air-gapped installs, custom
   CDNs); resolver verifies the SHA of whatever it returns too.

A mirror cannot tamper (hash mismatch fails hard) and cannot inflate (reads are
capped at `sizeBytes` plus a small slack); the worst a bad mirror can do is
fail over to the next URL.

## 6. Storage design (native)

- Layout: `<base>/flutter_litert/models/v1/<sha256>` (content-addressed; the
  layout version prefix allows future migrations). Original file name is
  recorded in a tiny sidecar JSON for debuggability.
- `<base>`: Application Support directory on Android, macOS, Windows, Linux
  (via path_provider). On iOS: Caches directory, because downloaded
  re-obtainable content must not enter iCloud backups (App Review guideline);
  Caches can be purged by the OS, which is acceptable since files re-download.
  This avoids needing native exclude-from-backup code in v1.
- Writes are atomic: stream to `<sha256>.part.<rand>`, hash incrementally
  during download (no second read), fsync, rename onto the final name only
  after the hash matches. A file that exists under its content hash name is
  complete and verified by construction; per-fetch revalidation is a cheap
  size check.
- Resume: on retry, if a `.part` exists and the server advertised
  `accept-ranges: bytes`, continue with a Range request (hash state restarts
  from the file, no re-download of received bytes).
- Concurrency: within an isolate, an in-flight map keyed by sha dedupes
  concurrent fetches. Across isolates/processes, atomic rename makes duplicate
  downloads safe (both produce identical files; last rename wins).
- Content addressing makes package upgrades safe: new model version means new
  hash means new cache entry. `evictExcept` garbage-collects superseded files.

## 7. Download engine (native)

- dart:io `HttpClient` (no package:http dependency), streaming, follows
  redirects, HTTPS-only enforced.
- Incremental SHA-256 via package:crypto (pure Dart) during streaming.
- Read cap at `sizeBytes` + slack; mismatch of declared vs received length is
  a failure.
- `RetryPolicy`: per-URL attempts (default 2) with exponential backoff and
  connect/idle timeouts, then next URL in the list.
- Progress callbacks receive cumulative bytes; `prefetch` aggregates across
  files.
- Honors `HTTP(S)_PROXY` via `HttpClient.findProxy` default behavior (note:
  environment-based, no custom API in v1).

## 8. Web

Default posture: web builds keep models as bundled assets (`platforms: [web]`),
so the resolver stops at step 1 and nothing here runs. For packages or apps
that remove assets entirely, ModelStore ships a web implementation:

- `fetch()` via browser fetch; **URLs must be CORS-enabled**. GitHub release
  URLs do not work (verified, section 3); `raw.githubusercontent.com` pinned to
  a tag works; self-hosted/CDN mirrors work if configured.
- Hashing via SubtleCrypto (`crypto.subtle.digest`), not pure-Dart crypto, to
  handle 50 MB+ files at native speed.
- Cache in CacheStorage (`flutter_litert_models_v1`), keyed by sha; in-memory
  fallback where CacheStorage is unavailable (e.g. some private modes). Quota
  failures surface as `ModelStorageException`.
- `CachedModel.filePath` is null; consumers use `readBytes()`. (LiteRT.js web
  paths already consume bytes.)

## 9. Platform matrix

| Platform | Transport | Hash | Cache | Notes |
|----------|-----------|------|-------|-------|
| Android  | dart:io   | crypto | App Support | Nothing special; INTERNET perm is default in Flutter templates |
| iOS      | dart:io   | crypto | Caches dir | ATS fine (HTTPS). Purgeable cache by design. Models are data, not executable code, so store-policy safe |
| macOS    | dart:io   | crypto | App Support | Sandboxed apps need `com.apple.security.network.client` entitlement; document loudly (AgeLapse already has it for other features) |
| Windows  | dart:io   | crypto | App Support | none |
| Linux    | dart:io   | crypto | App Support (XDG) | Flatpak needs `--share=network` (AgeLapse flatpak: verify) |
| Web      | fetch     | SubtleCrypto | CacheStorage | CORS constraints per section 8; default is bundled assets |

**Background isolates:** path_provider and rootBundle need
`BackgroundIsolateBinaryMessenger` setup inside spawned isolates. Guidance for
packages that run inference in isolates (cat/dog detectors do): resolve models
on the main isolate and pass file paths or bytes into the isolate, or construct
the isolate's ModelStore with an explicit `cacheDir` captured before spawn. The
resolver docs must call this out; it is the most likely integration mistake.

## 10. Hosting and manifest convention

- Each detection repo publishes model files as GitHub release assets under a
  **versioned, immutable tag**: `models-v<package major.minor>` (e.g.
  `models-v1.4`). Never a mutable `models` tag: replacing an asset under a tag
  that shipped breaks checksum verification for every old app in the field.
- Primary URL: the release asset. Backup URL baked into every spec:
  `raw.githubusercontent.com/<o>/<r>/<models-tag>/assets/models/<file>` (also
  the CORS-safe option). Both point at byte-identical content.
- Manifests are generated, not hand-typed: flutter_litert ships
  `dart run flutter_litert:gen_model_manifest --dir assets/models --url-base <release> --mirror-base <raw>`
  which hashes local files and emits a `model_manifest.dart` with the
  `ModelSpec` table. Regenerated whenever models change; CI can diff.
- Model license files ride along as release assets next to the models.

## 11. Per-package integration

| Package | Change | Version |
|---------|--------|---------|
| flutter_litert | Add `model_store.dart` library, manifest generator. New deps: crypto, path_provider | minor (3.6.0) |
| animal_detection | Delete asset declarations entirely (package has no web support, so `platforms: [web]` would be dead weight in the tarball); loads via ModelResolver; manifest | major |
| cat_detection / dog_detection | Same as animal_detection; drop own http/path_provider usage in favor of store | major |
| pose_detection | Assets -> `platforms: [web]` (package supports web); loads via ModelResolver; manifest | major |
| face_detection_tflite | Merge PR #12 seam (`loadModelBytes`); `ReleaseModelLoader` becomes a thin wrapper over ModelStore; assets stay bundled by default for now (core offline flow for AgeLapse); README documents how to unbundle | minor, then optional major later |
| hand_detection | AgeLapse should drop the dependency (only reads a version constant); package itself migrates like pose for other consumers | major |
| object_detection | Same pattern when touched | major |

AgeLapse changes: remove cat/dog/pose/animal/hand entries from all five strip
hacks (most of each list disappears); add first-use download UX (progress,
retry, offline error) when creating cat/dog/pose projects; optional settings
toggle for prefetch vs on-demand. Expected install-size effect: models drop
from ~213 MB to ~11 MB on every native platform.

### PR #12: merge first, refactor after

The contributor's public API (`ModelBytesLoader`, the `loadModelBytes`
parameter, `ReleaseModelLoader`) is exactly the surface the long-term design
keeps, so the PR can be merged substantially as-is and nothing merged now gets
reverted later. Sequence, deliberately decoupled from the ModelStore timeline
so the contributor is not kept waiting on a larger refactor (the PR has been
open since June 11):

1. **Merge PR #12 with authorship preserved** (merge or squash both retain the
   PR author as commit author). Mechanical review notes only: rebase onto
   current main (the PR targets 6.4.0; main is at 6.6.2, so it lands as
   6.7.0) and resolve the CHANGELOG accordingly. Credit him in the 6.7.0
   CHANGELOG entry ("thanks @<author>") and add a Contributors /
   Acknowledgements section to the README (one does not exist yet; only the
   original Python project is credited today). External contributions being
   visibly merged and credited is good signal for the repo.
2. **Follow-up maintainer commits, same or next release**, fixing the two
   known defects without touching his API: repoint `ReleaseModelLoader` and
   the SHA table at a versioned immutable `models-v*` tag (the current
   mutable `models` tag breaks old deployed versions if an asset is ever
   replaced), and document or fix the web path (GitHub release URLs are not
   fetchable cross-origin, per section 3; the raw mirror URL is).
3. **After flutter_litert 3.6.0 ships**: swap `ReleaseModelLoader` internals
   to delegate to `ModelStore.fetch` with face's generated manifest, and
   re-export the shared `ModelBytesLoader` typedef from flutter_litert.
   Public behavior unchanged; the downloader logic then lives in one place.

## 12. Rollout order

1. flutter_litert 3.6.0: ModelStore + resolver + generator + tests.
2. Publish `models-v*` release tags (and verify raw mirror URLs) for animal,
   cat, dog, pose, face, hand repos.
3. animal_detection major (cat/dog depend on it), then cat/dog/pose majors.
4. face_detection_tflite minor with reworked PR #12.
5. AgeLapse: bump deps, delete strip entries, add download UX.
6. Later, optionally: remove `platforms: [web]` assets from packages whose pub
   tarball size hurts (full removal, web via mirror), and publish optional
   `*_models` companion packages if any consumer asks for offline bundling
   without declaring assets themselves.

## 13. Alternatives considered and rejected

- **Per-package downloaders (PR #12 as-is, times seven):** duplicated logic,
  seven SHA tables, divergent cache layouts, each with its own bugs.
- **package:flutter_cache_manager:** no integrity verification, LRU eviction
  can silently evict a 70 MB model, brings sqflite, no suitable web story.
- **Play Asset Delivery / iOS On-Demand Resources:** best-in-class store
  integration but covers two platforms, requires deep per-app build changes,
  does nothing for Windows/Linux/web/sideloads. Not precluded later; the
  resolver's asset-first step composes with install-time packs.
- **Flutter deferred components:** Android-only, Play-only.
- **Config-driven build-time stripping via plugin build hooks:** flutter_litert
  could inject a Gradle strip task, but SwiftPM (which AgeLapse uses) cannot
  inject build phases on iOS/macOS, so it degenerates back into per-app
  scripts on Apple platforms. Kept as a documented recipe, not a mechanism.
- **Separate `litert_model_store` package:** keeps flutter_litert lean
  (crypto + path_provider stay out), but adds a publish/version surface and
  every detection package depends on flutter_litert anyway. Revisit only if
  the deps draw complaints; both are near-universal.

## 14. Risks and open questions

1. **Minimum Flutter version:** `platforms:` in assets requires Flutter >= 3.41
   (stable Nov 2025) in every consuming app. Acceptable for these packages
   (AgeLapse is on 3.44); it is still a floor raise for outside consumers and
   belongs in the CHANGELOGs. Packages could delay Half B and ship Half A only,
   but then binaries do not shrink.
2. **pub tarball size unchanged in v1:** with `platforms: [web]` the model
   files remain in the published package (pub.dev limit is 100 MB gzipped;
   cat/dog are within it today since they publish already). Full removal
   (section 12 step 6) is the lever if this becomes a problem.
3. **raw.githubusercontent.com as a serving host:** works and is CORS-safe, but
   large-file rate limits are undocumented. Mitigated by it being the backup
   (native primary is the release asset) and by `rewriteUrl`. Smoke-test a
   55 MB fetch before relying on it for web.
4. **iOS Caches purging:** a purge forces re-download at next use; acceptable,
   but the first-use UX in apps must tolerate "cached yesterday, gone today".
5. **Isolate integration mistakes** (section 9). Mitigate with docs plus a
   debug-mode error message that detects the missing messenger case.
6. **Supply chain:** hashes pinned in published packages, HTTPS-only, no
   "latest" URLs, generator makes hashes reproducible from local files. Tag
   deletion/re-push on GitHub is the residual risk; treat model tags as
   immutable by policy.
7. **Offline-first users:** the bundled path must remain first-class forever;
   asset-first resolution guarantees it, and docs must show the re-bundling
   recipe (declare the files as app assets).
8. **Who hosts object_detection-scale futures:** GitHub releases cap assets at
   2 GB each, fine for any tflite in sight.

## 15. Test plan

- Unit: hash verification (good/corrupt/truncated), mirror failover order,
  retry/backoff, dedupe (N concurrent fetches, one download), cancellation,
  eviction, seed-from-directory, URL rewrite.
- Fault injection: fake HttpClient factory (no network in unit tests).
- Integration (example app, per platform): real download from a `models-v*`
  tag, resume after kill mid-download, checksum mismatch path, offline
  failure UX, isolate usage pattern.
- Web integration: CORS fetch from raw mirror, CacheStorage persistence across
  reloads, SubtleCrypto hashing of a 50 MB file.
- Consumer-level: cat_detection example runs unbundled on all five native
  platforms; AgeLapse integration test creates a cat project end-to-end with
  an empty cache.
