// The play-services-litert prefab headers include this file, which Google's
// own CMake build generates from build_config.h.in. Its defaults (neither GPU
// nor NPU support compiled out) are reproduced here. Only the header-level
// feature macros depend on it; LiteRtRuntimeCApiStruct's layout does not.
#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_

#define LITERT_BUILD_CONFIG_DISABLE_GPU 0
#define LITERT_BUILD_CONFIG_DISABLE_NPU 0

#endif  // LITERT_BUILD_COMMON_BUILD_CONFIG_H_
