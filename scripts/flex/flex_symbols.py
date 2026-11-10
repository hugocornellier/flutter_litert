#!/usr/bin/env python3
"""Symbol checks for the iOS TensorFlowLiteFlex static archives.

Used by .github/workflows/build-flex-ios.yml. Compares a freshly built archive
with the previous release, checks the plugin entry points and the CocoaPods
nmedit hide list, and lists the symbols the archive expects someone else
(TensorFlowLiteC or the system) to provide.

Usage:
  flex_symbols.py --new-raw A --new-spm A --ref-raw A --ref-spm A \
      --tflc A --out DIR

All paths are device (ios-arm64) static archives:
  --new-raw  new TensorFlowLiteFlex.framework/TensorFlowLiteFlex
  --new-spm  new libTensorFlowLiteFlex.a (SPM form)
  --ref-raw  previous release raw archive (flex-v1.0.0)
  --ref-spm  previous release SPM archive (flex-v1.1.1)
  --tflc     TensorFlowLiteC archive the flex archive is linked with

Exits non-zero only on hard failures: a missing plugin entry point.
Everything else is reported.
"""

import argparse
import os
import re
import subprocess
import sys

PLUGIN = ["_tflite_plugin_create_delegate", "_tflite_plugin_destroy_delegate"]

# ios/flutter_litert.podspec: symbols hidden in TensorFlowLiteC (nmedit -R)
# when flutter_litert_flex is present.
NMEDIT_LIST = [
    "_TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider",
    "_TfLiteXNNPackDelegateCreate",
    "_TfLiteXNNPackDelegateCreateWithThreadpool",
    "_TfLiteXNNPackDelegateDelete",
    "_TfLiteXNNPackDelegateGetFlags",
    "_TfLiteXNNPackDelegateGetOptions",
    "_TfLiteXNNPackDelegateGetThreadPool",
    "_TfLiteXNNPackDelegateInMemoryFilePath",
    "_TfLiteXNNPackDelegateOptionsDefault",
    "_TfLiteXNNPackDelegateWeightsCacheCreate",
    "_TfLiteXNNPackDelegateWeightsCacheCreateWithSize",
    "_TfLiteXNNPackDelegateWeightsCacheDelete",
    "_TfLiteXNNPackDelegateWeightsCacheFinalizeHard",
    "_TfLiteXNNPackDelegateWeightsCacheFinalizeSoft",
    "__ZN6tflite19AcquireFlexDelegateEv",
]

C_API = re.compile(r"^_(TfLite|tflite_|TF_|TFE_)")
TFLITE_ISH = re.compile(
    r"TfLite|tflite|6tflite|xnn|XNN|pthreadpool|cpuinfo|3ruy|gemmlowp|"
    r"flatbuffers|farmhash|fp16_ieee|_Z.*4absl"
)


def nm_lines(args, path):
    out = subprocess.run(
        ["xcrun", "nm"] + args + [path],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    return out.splitlines()


def defined(path):
    """Return (all defined globals, strong defined globals)."""
    every, strong = set(), set()
    for line in nm_lines(["-g", "-U", "-m"], path):
        if " external " not in line:
            continue
        # "external" and "private external" (hidden visibility) both take
        # part in a static link, so both can collide.
        head, tail = line.split(" external ", 1)
        name = tail.split(" (", 1)[0].strip()
        if not name:
            continue
        every.add(name)
        if "weak" not in head:
            strong.add(name)
    return every, strong


def undefined(path):
    names = set()
    for line in nm_lines(["-u", "-j"], path):
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        names.add(line.split()[-1])
    return names


def write(out, name, items):
    with open(os.path.join(out, name), "w") as f:
        for s in sorted(items):
            f.write(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    for k in ("new-raw", "new-spm", "ref-raw", "ref-spm", "tflc", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("== reading symbol tables")
    nr_all, nr_strong = defined(a.new_raw)
    ns_all, ns_strong = defined(a.new_spm)
    rr_all, _ = defined(a.ref_raw)
    rs_all, _ = defined(a.ref_spm)
    tc_all, tc_strong = defined(a.tflc)
    ns_undef = undefined(a.new_spm)
    nr_undef = undefined(a.new_raw)
    rs_undef = undefined(a.ref_spm)
    rs_all_set = rs_all
    write(a.out, "defined-tflc.txt", tc_all)

    hard_fail = False

    print("\n== plugin entry points (defined, global)")
    for name, syms in (("new raw", nr_all), ("new spm", ns_all)):
        for p in PLUGIN:
            ok = p in syms
            print(f"  {name}: {p}: {'present' if ok else 'MISSING'}")
            hard_fail |= not ok

    print("\n== defined global symbols vs previous release")
    for label, new, ref, tag in (
        ("raw (vs flex-v1.0.0)", nr_all, rr_all, "raw"),
        ("spm (vs flex-v1.1.1)", ns_all, rs_all, "spm"),
    ):
        added, removed = new - ref, ref - new
        write(a.out, f"defined-{tag}-new.txt", new)
        write(a.out, f"defined-{tag}-ref.txt", ref)
        write(a.out, f"defined-{tag}-added.txt", added)
        write(a.out, f"defined-{tag}-removed.txt", removed)
        print(f"  {label}: new {len(new)}, ref {len(ref)}, "
              f"added {len(added)}, removed {len(removed)}")
        ca, cr = sorted(s for s in added if C_API.match(s)), sorted(
            s for s in removed if C_API.match(s))
        print(f"    C API added ({len(ca)}): {' '.join(ca[:40])}")
        print(f"    C API removed ({len(cr)}): {' '.join(cr[:40])}")

    print("\n== podspec nmedit hide list (15 symbols)")
    print("  symbol | defined in new raw | in new spm | in TFLiteC | strong in both raw and TFLiteC")
    for s in NMEDIT_LIST:
        print(f"  {s} | {s in nr_all} | {s in ns_all} | {s in tc_all} | "
              f"{s in nr_strong and s in tc_strong}")
    overlap_raw = nr_strong & tc_strong
    overlap_spm = ns_strong & tc_strong
    write(a.out, "overlap-raw-tflc.txt", overlap_raw)
    write(a.out, "overlap-spm-tflc.txt", overlap_spm)
    extra = sorted(overlap_raw - set(NMEDIT_LIST))
    gone = sorted(set(NMEDIT_LIST) - overlap_raw)
    print(f"  strong overlap raw x TFLiteC: {len(overlap_raw)} "
          f"(not in list: {len(extra)}, list entries not overlapping: {len(gone)})")
    print(f"    not in list: {' '.join(extra[:60])}")
    print(f"    list entries not overlapping: {' '.join(gone)}")
    print(f"  strong overlap spm x TFLiteC: {len(overlap_spm)}: "
          f"{' '.join(sorted(overlap_spm)[:60])}")

    print("\n== symbols the SPM archive expects from outside")
    for label, undef, defs, tag in (
        ("new spm", ns_undef, ns_all, "spm-new"),
        ("new raw", nr_undef, nr_all, "raw-new"),
        ("ref spm", rs_undef, rs_all_set, "spm-ref"),
    ):
        ext = undef - defs
        tfl = {s for s in ext if TFLITE_ISH.search(s)}
        other = ext - tfl
        from_tc = tfl & tc_all
        not_tc = tfl - tc_all
        write(a.out, f"external-{tag}-tflite.txt", tfl)
        write(a.out, f"external-{tag}-tflite-not-in-tflc.txt", not_tc)
        write(a.out, f"external-{tag}-other.txt", other)
        print(f"  {label}: {len(ext)} external; TFLite-like {len(tfl)} "
              f"(provided by the TFLiteC given: {len(from_tc)}, "
              f"not provided: {len(not_tc)}); other {len(other)}")
        if tag != "spm-ref":
            print(f"    not provided: {' '.join(sorted(not_tc)[:80])}")
    new_ext = (ns_undef - ns_all)
    ref_ext = (rs_undef - rs_all_set)
    newly = sorted(s for s in new_ext - ref_ext if TFLITE_ISH.search(s))
    write(a.out, "external-spm-tflite-newly-required.txt", newly)
    print(f"  TFLite-like externals new in 2.21 SPM vs 2.20 SPM: {len(newly)}: "
          f"{' '.join(newly[:80])}")

    if hard_fail:
        print("::error::plugin entry point missing")
        sys.exit(1)


if __name__ == "__main__":
    main()
