"""Make a TensorFlow Lite CMake build report the real TF version.

From TF v2.21.0 on, tensorflow/lite/version.h no longer includes
release_version.h. It defaults TF_MINOR_VERSION to 19, and
tensorflow/lite/CMakeLists.txt adds the PRIVATE compile definition
TF_MINOR_VERSION=19 to the tensorflow-lite target. A CMake build of
tensorflowlite_c would then report 2.19.0 from TfLiteVersion() even when it
is built from the v2.21.0 sources.

This script rewrites both places to the given major.minor.patch, so every
translation unit (tensorflow-lite and tensorflowlite_c) agrees.

Usage: python patch_tflite_cmake_version.py <tf-src-dir> <major.minor.patch>
"""

import pathlib
import sys


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        sys.exit(f"{path}: expected exactly one {old!r}, found {count}")
    path.write_text(text.replace(old, new))
    print(f"{path}: {old.strip()!r} -> {new.strip()!r}")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root = pathlib.Path(sys.argv[1])
    major, minor, patch = sys.argv[2].split(".")

    cmake = root / "tensorflow/lite/CMakeLists.txt"
    replace_once(cmake, "    TF_MAJOR_VERSION=2\n", f"    TF_MAJOR_VERSION={major}\n")
    replace_once(cmake, "    TF_MINOR_VERSION=19\n", f"    TF_MINOR_VERSION={minor}\n")
    replace_once(cmake, "    TF_PATCH_VERSION=0\n", f"    TF_PATCH_VERSION={patch}\n")

    version_h = root / "tensorflow/lite/version.h"
    replace_once(version_h, "#define TF_MAJOR_VERSION 2\n", f"#define TF_MAJOR_VERSION {major}\n")
    replace_once(version_h, "#define TF_MINOR_VERSION 19\n", f"#define TF_MINOR_VERSION {minor}\n")
    replace_once(version_h, "#define TF_PATCH_VERSION 0\n", f"#define TF_PATCH_VERSION {patch}\n")


if __name__ == "__main__":
    main()
