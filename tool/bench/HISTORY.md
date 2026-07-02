# Micro-benchmark history

Method: `tool/bench/run_ab.sh` compiles the base ref and the working tree to
separate AOT binaries and runs them interleaved (A B A B ...) from the repo
root, so both sides share dylibs, models, and machine state. Verdicts use the
median delta with a seeded bootstrap 95% CI (`compare.dart`). All numbers are
macOS arm64 AOT; on-device (Android/iOS profile) validation is a separate
step.

Acceptance rule: a change lands only if its within-run median delta clears
the A/A noise floor and the test suite passes. Anything inside the floor is
reverted, even when it looks positive.

## Noise floor (A/A, 2026-07-01, commit 9ef761c)

Two independent AOT builds of identical source, 6 interleaved rounds:
deltas up to 2.2% with CIs excluding zero (build layout noise; samples within
one binary are correlated, so the bootstrap cannot see rebuild variance).
Floor set at 3% median delta on the affected bench. Absolute medians also
drift up to ~8% across sessions; only within-run interleaved deltas count.

## Accepted (2026-07-01)

- a0b39e2 perf(tensor): stop building error message strings on the happy
  path. interp_add_f32 760 -> 450 ns/op (-40.8%). quiver checkState/
  checkArgument evaluate `message:` eagerly; setTo and copyTo interpolated
  multi-int and runtimeType strings per successful call.
- bb11740 perf(tensor): derive typed-data shape check from tensor byte size.
  interp_add_f32 457 -> 423 ns/op (-7.4%). Was NumDims + one Dim FFI call
  per axis plus list allocations per run.
- 5c76249 perf(interpreter): skip tensor list rebuild on the steady-state
  run path. interp_add_f32 429 -> 404 ns/op (-5.7%). Fresh pointer per
  index; fused shape-check + setTo pass; resize/allocate falls back to the
  two-pass flow. Fresh-pointer-per-run contract from 927480e preserved.
- 409936b perf(interpreter): single-output fast path for run().
  interp_add_f32 400 -> 364 ns/op (-9.0%). Skips the one-entry output map.
- 0b63bd8 perf(compiled-model): stop building lock label strings on the
  happy path. cm_add_managed_run 717 -> 525 ns/op (-26.8%). Managed runs
  built six interpolated lock/unlock label strings per call; _checkAt now
  takes the pieces and builds the label only when throwing.

Net vs 9ef761c: interp_add_f32 771 -> 364 ns/op (-52.8%),
cm_add_managed_run 736 -> 541 ns/op (-26.5%). Real-model paths (face
detection, ~1 ms) unchanged, as expected for fixed few-hundred-ns per-call
savings. hostMemory paths unchanged.

## Rejected (kept for the record; do not re-try without new evidence)

- Cache Float32List views over CompiledModel host memory buffers:
  cm_add_host_zero -1.9% (about -8 ns), below floor. asTypedList wrapper
  costs only a few ns.
- isLeaf on LiteRtUnlockTensorBuffer + LiteRtHasTensorBufferEvent: no
  measurable effect (managed-path cost is inside the native lock/unlock
  functions, not the FFI transition). lockTensorBuffer must stay non-leaf
  anyway: GPU-backed locks can block on a fence and a blocking leaf call
  stalls GC safepoints.
- Cache Tensor.type as late final: -2.2%, below floor.
- Direct Pointer<Float>.asTypedList instead of Uint8 view + buffer +
  Float32List.view in _withLockedFloats: -2.0%, below floor.

## Async dispatch (2026-07-01)

- 0c295b7 test(bench): cm_*_async_run benches measure awaited latency plus
  worst event-loop stall (1 ms ticker; starved windows count the pending
  gap).
- 7f8a860 perf(compiled-model): run async dispatch on a helper isolate.
  Before: runAsync blocked the calling isolate inside the native call, so
  a 4 ms burst of awaited runs starved the event loop for the whole window
  (2.9-3.0 ms worst stall). After: blocking LiteRtRunCompiledModel runs on
  a lazily-spawned per-model helper isolate; stall drops 95% to ~142 us
  (ticker granularity plus scheduler jitter, not blocking). Cost: one
  isolate message round trip per dispatch: +3.0 us awaited latency on the
  tiny add model, +0.62% on the face model. Sync run() unchanged (all
  benches within the noise floor). runAsync serializes FIFO; overlapping
  bare dispatchAsync throws; close() during an in-flight dispatch throws.

Guidance: prefer run() for very fast models when blocking is fine; use
runAsync when the isolate must stay responsive (UI isolate inference,
pipeline overlap) or the model is slow enough that ~3 us is noise.

## Shared detection utilities (2026-07-01)

- 8e9c4e0 test(bench): utility benches. JIT baseline: bgr norm ~112 us,
  rgba norm ~95 us, 720p YUV pack ~209 us, YOLO decode ~702 us, NMS
  ~10.6 us; about 1 ms of Dart per frame combined.
- 8818d65 perf(model-output): logit-space threshold prune before sigmoid.
  util_postprocess_yolo -3.0/-3.1% across two confirming runs. Exact.
- 322ab54 perf(model-output): SIMD (Float32x4) argmax for channel-major
  decode. util_postprocess_yolo 638 -> 187 us/op (-70.7%). Layout
  equivalence tests added (SIMD vs scalar vs anchor-major, exact ties).
- bb65762 perf(image-tensor): 256-entry LUT normalization, bit-identical.
  util_bgr_signed_f32 -4.2%; rgba path neutral but unified on the LUT.
- 9ba5ed7 perf(yuv): optional `into` reuse buffer for packYuv420.
  Reuse path 29.8 us vs allocating 114.4 us AOT (-74%, opt-in); output
  byte-identical (truncated planes now zero-fill their tails).

Net vs 8e9c4e0 (6 interleaved rounds): util_postprocess_yolo -72.3%
(682 -> 189 us), util_bgr_signed_f32 -4.2%, util_yuv_pack_720p -3.1%,
everything else including inference and async paths unchanged.

Rejected: channel-major argmax loop inversion (scalar), +7% SLOWER;
consecutive anchors share cache lines so the strided loop was already
cache-friendly, and the running max moved from register to memory.
Not pursued: NMS micro-tuning (~10 us absolute, low value); the legacy
Map-based postProcessDetections path (callers should move to
postProcessDetectionsFlat instead).

Downstream one-liners (per repo, optional): pass a persistent `into`
buffer to packYuv420 in live-camera loops; hand_detection still calls
fillNHWC4DFromBgrBytes (nested-list path) in one place; pose_detection
still calls legacy postProcessDetections/decodeAndSplitOutputs in one
place.

## Branch audit (2026-07-02)

Every commit re-reviewed (inline pass plus an adversarial multi-agent
review of the full diff; 52 raw candidates triaged), every claim
re-measured on fresh interleaved desktop runs, full test suite green,
and a physical-iPhone profile-mode A/B (branch tip vs 9ef761c) run via
example/integration_test/perf_bench_test.dart.

Fixes that came out of the audit (ec430a6):
- Sync run/dispatch/writeInput/readOutput now throw StateError while an
  async dispatch is in flight (the helper isolate removed the mutual
  exclusion blocking dispatch used to provide implicitly), and a queued
  runAsync can no longer overwrite buffers a bare dispatchAsync is
  running against.
- postProcessDetectionsFlat validates out.length up front (the SIMD
  path reads through out.buffer and could silently read past a short or
  view-backed buffer).
- Docs: runAsync input-snapshot timing, dispatchAsync completion
  semantics, and the unvalidated thread-affine mobile GPU caveat.
Guard overhead measured within the noise floor.

Desktop re-verification: interp_add_f32 -52.1%, cm_add_managed -27.5%
(vs 9ef761c, 8 rounds); util_postprocess_yolo -72.4%,
util_bgr_signed_f32 -3.6% (vs 8e9c4e0); async stall -95.0/-95.3% (vs
0c295b7). All previously-unchanged benches still within the floor.

iPhone (profile, iOS 26.5, single run per side, ns/op medians):
- interp_add_f32 878 -> 393 (-55%)
- cm_add_managed_run 941 -> 681 (-28%)
- cm_add_host_run 606 -> 589 (flat, as designed)
- util_postprocess_yolo 762k -> 225k (-70%, NEON SIMD confirmed)
- util_bgr_signed_f32 142k -> 137k (-4%)
- util_yuv_pack_720p flat (allocating path, as designed)
- cm_add_async_run stall 2.93 ms -> 194 us (-93%); awaited cost
  1.1 us -> 25.5 us (isolate messaging is pricier on iOS than macOS;
  ~4% of a face-sized model, the responsiveness win is the point)
Also the first successful CompiledModel validation on a physical
iPhone, including helper-isolate runAsync.

Review candidates deliberately not actioned (recorded so they are not
relitigated): three TensorType byte-width tables could consolidate;
_AsyncDispatcher could keep one persistent reply port (saves ~us per
dispatch, more attractive given iOS messaging cost); SIMD scratch
buffers could be caller-supplied; run() fast path intentionally
duplicates runForMultipleInputs' tail (measured -9%); packYuv420 kept
`into` naming because its throw-on-wrong-size contract differs from the
image utils' allocate-if-absent `buffer`; bench harness and the device
test intentionally duplicate the measurement core to stay self-contained
per revision.

## Open opportunity (owner decision, not a micro-tune)

After the label fix, managed mode's remaining ~70 ns premium over
hostMemory is four native lock/unlock calls per run. Closing it means
locking CPU host-memory-backed managed buffers once and caching the host
pointer, which bends the documented lock/unlock-per-access contract, or
defaulting CPU-only models to TensorBufferMode.hostMemory, which changes
which methods are callable on default models. Both need a deliberate call.

## Known non-candidates

- Caching Tensor wrappers across runs: forbidden; fresh pointers per run fix
  a real native crash (see 927480e, XNNPACK storage relocation).
- SignatureRunner name churn: already fixed via name-keyed tensor caches in
  signature_runner.dart.
