#!/usr/bin/env python3
"""Checks the native libraries an APK packages for its flutter_litert runtime.

  python3 -I scripts/check_android_runtime_apk.py --runtime playServices app.apk
  python3 -I scripts/check_android_runtime_apk.py --runtime bundled app.apk
  python3 -I scripts/check_android_runtime_apk.py --runtime auto app.apk

For flutterLitert.runtime=playServices, every Flutter ABI in the APK must carry
the runtime shim and the two Play services client libraries, and none of the
LiteRT libraries the bundled mode packages. The shim must export each LiteRt
function lib/src/bindings/litert_ffi.dart binds, and nothing else beyond its
own API, and depend on the Play services client.

For the bundled default, the APK must carry libLiteRt.so, which serves both
Interpreter and CompiledModel, the GPU accelerator on arm64-v8a and x86_64 and
not on armeabi-v7a (pass --no-gpu-accelerator for an app built with
flutterLitert.bundleGpuAccelerator=false), and a shim that neither forwards
LiteRt nor links Play services.

For auto, the APK must carry both: the Play services shim and clients, and the
bundled libraries.

In every mode the APK must not carry a LiteRT 1.x library
(libtensorflowlite_jni.so, libtensorflowlite_gpu_jni.so,
libtensorflowlite_flex_jni.so) or the Java of the litert 1.x AARs, which does
not fit the litert-api that play-services-litert brings. In bundled mode no
org.tensorflow.lite class may be packaged at all, since nothing in that mode
brings one. R8 can hide that Java in a release APK, so check a debug APK as
well.

Only reads the APK; nothing in it is executed.
"""

import argparse
import pathlib
import re
import struct
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DART_BINDINGS = ROOT / "lib" / "src" / "bindings" / "litert_ffi.dart"

SHIM = "libflutter_litert_runtime.so"
PLAY_SERVICES_CLIENTS = (
    "liblitert_compiledmodel_gms_client.so",
    "libtensorflowlite_jni_gms_client.so",
)
# The bundled runtime: one library for Interpreter and CompiledModel.
BUNDLED_RUNTIME = "libLiteRt.so"
GPU_ACCELERATOR = "libLiteRtClGlAccelerator.so"
# The ABIs the bundled GPU accelerator is packaged for. litert-gpu also ships
# one for armeabi-v7a, which the plugin leaves out on purpose.
GPU_ACCELERATOR_ABIS = ("arm64-v8a", "x86_64")
# Libraries that only the bundled mode downloads and packages.
BUNDLED_ONLY = (
    BUNDLED_RUNTIME,
    GPU_ACCELERATOR,
    "libLiteRtOpenClAccelerator.so",
    "libLiteRtWebGpuAccelerator.so",
)
# LiteRT 1.x libraries, which no runtime mode packages any more.
LITERT_1X_LIBRARIES = (
    "libtensorflowlite_jni.so",
    "libtensorflowlite_gpu_jni.so",
    "libtensorflowlite_flex_jni.so",
)
# Classes only the litert and litert-gpu 1.x AARs have, as dex stores their
# names. The litert-api interfaces that play-services-litert brings share
# their package, so playServices and auto check these names only.
LITERT_1X_JAVA_CLASSES = (
    b"Lorg/tensorflow/lite/NativeInterpreterWrapper;",
    b"Lorg/tensorflow/lite/gpu/GpuDelegate;",
)
LITERT_JAVA_PACKAGE = b"Lorg/tensorflow/lite/"
SHIM_EXPORTS = (
    "FlutterLiteRtRuntimeMode",
    "FlutterLiteRtRuntimeSelected",
    "FlutterLiteRtRuntimeSelect",
    "FlutterLiteRtRuntimePlayServicesError",
    "FlutterLiteRtRuntimeInterpreterState",
    "FlutterLiteRtRuntimeCompiledModelState",
    "FlutterLiteRtRuntimeInterpreterError",
    "FlutterLiteRtRuntimeCompiledModelError",
    "FlutterLiteRtRuntimeGpuError",
    "FlutterLiteRtRuntimeApi",
    "FlutterLiteRtRuntimeGpuAccelerator",
    "FlutterLiteRtRuntimeContext",
    "Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeSetInterpreterState",
)
# JNI entry points only the Play services PlayServicesRuntime.java declares.
PLAY_SERVICES_SHIM_EXPORTS = (
    "Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeAttachCompiledModel",
    "Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeSetCompiledModelFailure",
)

SHT_DYNSYM = 11
SHT_DYNAMIC = 6
DT_NEEDED = 1
SHN_UNDEF = 0
STB_GLOBAL = 1
STB_WEAK = 2
STV_DEFAULT = 0


def dart_bound_litert_functions():
    names = set(re.findall(r">\(\s*'(LiteRt\w+)'\s*\)", DART_BINDINGS.read_text()))
    if not names:
        sys.exit(f"no LiteRt lookups found in {DART_BINDINGS}")
    return names


def elf64_dynamic_info(data):
    """Returns (exported symbol names, DT_NEEDED names) of a 64-bit LE ELF."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]

    def strtab(index, offset):
        base = sections[index][4]
        end = data.index(b"\0", base + offset)
        return data[base + offset : end].decode()

    exports = set()
    needed = []
    for _, sh_type, _, _, offset, size, link, _, _, entsize in sections:
        if sh_type == SHT_DYNSYM:
            for pos in range(offset, offset + size, entsize):
                name, info, other, shndx = struct.unpack_from("<IBBH", data, pos)
                bind = info >> 4
                if (
                    shndx != SHN_UNDEF
                    and bind in (STB_GLOBAL, STB_WEAK)
                    and other & 3 == STV_DEFAULT
                    and name
                ):
                    exports.add(strtab(link, name))
        elif sh_type == SHT_DYNAMIC:
            for pos in range(offset, offset + size, entsize):
                tag, value = struct.unpack_from("<qQ", data, pos)
                if tag == DT_NEEDED:
                    needed.append(strtab(link, value))
    return exports, needed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runtime", choices=("bundled", "playServices", "auto"), required=True
    )
    parser.add_argument(
        "--no-gpu-accelerator",
        action="store_true",
        help="the app was built with flutterLitert.bundleGpuAccelerator=false",
    )
    parser.add_argument("apk", type=pathlib.Path)
    args = parser.parse_args()
    # Whether the Play services client is linked, and LiteRT itself packaged.
    play = args.runtime != "bundled"
    bundled = args.runtime != "playServices"
    if args.no_gpu_accelerator and not bundled:
        parser.error("--no-gpu-accelerator only applies to bundled and auto")

    with zipfile.ZipFile(args.apk) as apk:
        libs = {}
        sizes = {}
        for info in apk.infolist():
            parts = info.filename.split("/")
            if len(parts) == 3 and parts[0] == "lib" and parts[2].endswith(".so"):
                libs.setdefault(parts[1], {})[parts[2]] = info.filename
                sizes[info.filename] = info.file_size
        # Dependencies can ship ABIs Flutter does not build for (x86); only
        # the ones holding the Flutter engine are part of the app.
        libs = {abi: files for abi, files in libs.items() if "libflutter.so" in files}
        if not libs:
            sys.exit(f"{args.apk}: no Flutter native libraries")

        errors = []
        for name in apk.namelist():
            if not name.endswith(".dex"):
                continue
            dex = apk.read(name)
            for java_class in LITERT_1X_JAVA_CLASSES:
                if java_class in dex:
                    errors.append(
                        f"{name}: must not link the litert 1.x Java "
                        f"({java_class.decode()})"
                    )
            if not play and LITERT_JAVA_PACKAGE in dex:
                errors.append(f"{name}: bundled must not package org.tensorflow.lite Java")
        litert_functions = dart_bound_litert_functions()
        for abi, files in sorted(libs.items()):
            present = set(files)
            required = {SHIM}
            forbidden = set(LITERT_1X_LIBRARIES)
            if play:
                required |= set(PLAY_SERVICES_CLIENTS)
            else:
                forbidden |= set(PLAY_SERVICES_CLIENTS)
            if bundled:
                required.add(BUNDLED_RUNTIME)
                if not args.no_gpu_accelerator and abi in GPU_ACCELERATOR_ABIS:
                    required.add(GPU_ACCELERATOR)
                else:
                    forbidden.add(GPU_ACCELERATOR)
            else:
                forbidden |= set(BUNDLED_ONLY)
            for lib in sorted(required - present):
                errors.append(f"{abi}: missing {lib}")
            for lib in sorted(forbidden & present):
                errors.append(f"{abi}: must not package {lib}")
            total = sum(sizes[files[lib]] for lib in present)
            print(f"{abi}: {', '.join(sorted(present))}")
            print(f"{abi}: {total} bytes of native libraries, uncompressed")

            if SHIM not in present or abi not in ("arm64-v8a", "x86_64"):
                continue
            exports, needed = elf64_dynamic_info(apk.read(files[SHIM]))
            shim_exports = SHIM_EXPORTS + (PLAY_SERVICES_SHIM_EXPORTS if play else ())
            for symbol in shim_exports:
                if symbol not in exports:
                    errors.append(f"{abi}: {SHIM} does not export {symbol}")
            forwarded = {s for s in exports if s.startswith("LiteRt")}
            client_linked = "liblitert_compiledmodel_gms_client.so" in needed
            if play:
                for symbol in sorted(litert_functions - forwarded):
                    errors.append(f"{abi}: {SHIM} does not forward {symbol}")
                for symbol in sorted(forwarded - litert_functions):
                    errors.append(f"{abi}: {SHIM} exports unbound {symbol}")
                if not client_linked:
                    errors.append(f"{abi}: {SHIM} does not link the Play services client")
                print(f"{abi}: {SHIM} forwards {len(forwarded)} LiteRt functions")
            else:
                if forwarded:
                    errors.append(f"{abi}: bundled {SHIM} exports {sorted(forwarded)}")
                if client_linked:
                    errors.append(f"{abi}: bundled {SHIM} links the Play services client")
            unexpected = sorted(
                s for s in exports
                if s not in shim_exports and not s.startswith("LiteRt")
            )
            if unexpected:
                errors.append(f"{abi}: {SHIM} leaks symbols {unexpected}")

    if errors:
        print("\n".join(["FAILED:"] + errors), file=sys.stderr)
        return 1
    print(f"OK: {args.apk} packages the {args.runtime} runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
