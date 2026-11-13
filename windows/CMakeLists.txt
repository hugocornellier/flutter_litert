cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "flutter_tflite")
project(${PROJECT_NAME} LANGUAGES CXX C)

cmake_policy(VERSION 3.14...3.25)

set(PLUGIN_NAME "flutter_tflite_plugin")

list(APPEND PLUGIN_SOURCES
  "flutter_tflite_plugin.cpp"
  "flutter_tflite_plugin.h"
)

add_library(${PLUGIN_NAME} SHARED
  "include/flutter_tflite/flutter_tflite_plugin_c_api.h"
  "flutter_tflite_plugin_c_api.cpp"
  ${PLUGIN_SOURCES}
)

apply_standard_settings(${PLUGIN_NAME})

# Symbols are hidden by default to reduce the chance of accidental conflicts
# between plugins. Any symbols that should be exported must use FLUTTER_PLUGIN_EXPORT.
set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

# Build the TFLite custom ops library (e.g., Convolution2DTransposeBias for MediaPipe)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/custom_ops")

# ON (the default) bundles this plugin's CompiledModel runtime: libLiteRt.dll,
# the WebGPU accelerator and the DirectX Shader Compiler DLLs. An app that
# ships another copy of the LiteRT runtime under the same file names (LiteRT-LM
# does) can turn it OFF, so the files are neither downloaded nor bundled. The
# TensorFlow Lite C library (Interpreter), the custom ops library and
# vcruntime140_threads.dll are bundled in both settings.
# An app sets it in its own windows/CMakeLists.txt, before the line
# include(flutter/generated_plugins.cmake):
#   set(FLUTTER_LITERT_BUNDLE_DESKTOP_RUNTIME OFF)
option(FLUTTER_LITERT_BUNDLE_DESKTOP_RUNTIME
  "Bundle flutter_litert's CompiledModel runtime (libLiteRt, the WebGPU accelerator and DXC)"
  ON)

set(FLUTTER_LITERT_TFLITE_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libtensorflowlite_c-win.dll")
# LiteRT Next runtime (CompiledModel API), x86_64; from the official
# ai-edge-litert 2.3.0 PyPI wheel.
set(FLUTTER_LITERT_LITERT_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRt.dll")
# WebGPU (Dawn/D3D12) GPU accelerator for the CompiledModel API; from the same
# ai-edge-litert 2.3.0 wheel, unmodified. It must match libLiteRt.dll: the
# runtime/accelerator ABI changed between 2.1.5 and 2.3.0. The runtime
# discovers it at startup by scanning the RuntimeLibraryDir environment
# option for this exact file name, so the name must not change.
set(FLUTTER_LITERT_GPU_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libLiteRtWebGpuAccelerator.dll")
# DirectX Shader Compiler (DXC), from microsoft/DirectXShaderCompiler release
# v1.9.2602.24. Optional but preferred: Dawn uses DXC instead of the OS FXC
# (d3dcompiler_47.dll) when both dxcompiler.dll and dxil.dll are present next
# to the executable.
set(FLUTTER_LITERT_DXCOMPILER_DLL "${CMAKE_CURRENT_SOURCE_DIR}/dxcompiler.dll")
set(FLUTTER_LITERT_DXIL_DLL "${CMAKE_CURRENT_SOURCE_DIR}/dxil.dll")
# Microsoft's app-local redist copy of vcruntime140_threads.dll (C11 threads,
# VC++ 14.38+), from Microsoft.VC143.CRT 14.44.35211.0, unmodified (sha256
# 219915cf20822f34d5e7c1fdd4e21ae7f3396881096c51036225fb8f84b47afa).
# libLiteRt.dll (CompiledModel) imports it, and it is not among the VC++
# runtime DLLs Flutter apps deploy app-locally, so PCs without a recent VC++
# Redistributable would fail to load libLiteRt.dll. libtensorflowlite_c-win.dll
# is built with the static CRT and needs no VC++ runtime DLL.
# It stays bundled when FLUTTER_LITERT_BUNDLE_DESKTOP_RUNTIME is OFF: the
# flutter_litert_flex Windows library imports it too.
set(FLUTTER_LITERT_VCRUNTIME_THREADS_DLL "${CMAKE_CURRENT_SOURCE_DIR}/vcruntime140_threads.dll")

# The WebGPU GPU accelerator (~30 MB) and DirectX Shader Compiler (~18 MB) are
# fetched at build time from a GitHub release rather than bundled in the
# published pub.dev package, which must stay under the 100 MiB limit. Each is
# downloaded (verified by SHA-256) only if it is not already present with the
# pinned hash. dxcompiler.dll and dxil.dll are the same files as in the
# litert-desktop-gpu-v1.0.0 release.
set(FLUTTER_LITERT_GPU_LIBS_URL
  "https://github.com/hugocornellier/flutter_litert/releases/download/litert-desktop-gpu-v2.3.0")

function(flutter_litert_fetch_lib path url sha)
  # A file left by an older flutter_litert (a different pinned version) is
  # replaced, not reused: the accelerator must match the bundled runtime.
  if(EXISTS "${path}")
    file(SHA256 "${path}" existing_sha)
    if(existing_sha STREQUAL sha)
      return()
    endif()
    message(STATUS "flutter_litert: ${path} does not match the pinned SHA-256; replacing it")
    file(REMOVE "${path}")
  endif()
  # GitHub answers a release download with an error now and then, for seconds
  # at a time: three attempts made back to back all failed in a Windows CI run
  # on 2026-11-05. So the attempts are spaced 5, 10, 20 and 40 s apart, and a
  # download that stalls for 60 s counts as failed instead of hanging the
  # build. The hash is checked here rather than with EXPECTED_HASH: a failed
  # EXPECTED_HASH download is a CMake error, which fails the configure step at
  # its end even when a later attempt succeeds.
  set(delay 5)
  foreach(attempt RANGE 1 5)
    message(STATUS "flutter_litert: downloading ${url} (attempt ${attempt})")
    file(DOWNLOAD "${url}" "${path}.part"
      TLS_VERIFY ON
      INACTIVITY_TIMEOUT 60
      STATUS dl_status)
    list(GET dl_status 0 dl_code)
    if(dl_code EQUAL 0)
      file(SHA256 "${path}.part" dl_sha)
      if(dl_sha STREQUAL sha)
        file(RENAME "${path}.part" "${path}")
        return()
      endif()
      set(dl_msg "SHA-256 ${dl_sha}, expected ${sha}")
    else()
      list(GET dl_status 1 dl_msg)
    endif()
    file(REMOVE "${path}.part")
    if(attempt LESS 5)
      message(WARNING "flutter_litert: download failed (${dl_msg}); retrying in ${delay} s")
      execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep ${delay})
      math(EXPR delay "${delay} * 2")
    else()
      message(WARNING "flutter_litert: download failed (${dl_msg})")
    endif()
  endforeach()
  message(FATAL_ERROR "flutter_litert: could not download ${url}")
endfunction()

set(flutter_litert_bundled_libraries
  "${FLUTTER_LITERT_TFLITE_DLL}"
  "${FLUTTER_LITERT_VCRUNTIME_THREADS_DLL}"
  "$<TARGET_FILE:tflite_custom_ops>"
)

if(FLUTTER_LITERT_BUNDLE_DESKTOP_RUNTIME)
  flutter_litert_fetch_lib("${FLUTTER_LITERT_GPU_DLL}"
    "${FLUTTER_LITERT_GPU_LIBS_URL}/libLiteRtWebGpuAccelerator.dll"
    "1cf8b23cd30d04f0c7f087243322a4550a695c4803be9b5ef5027d7de3e26d8d")
  flutter_litert_fetch_lib("${FLUTTER_LITERT_DXCOMPILER_DLL}"
    "${FLUTTER_LITERT_GPU_LIBS_URL}/dxcompiler.dll"
    "9b5e10ed756c461b4ec2c83a99f1d6ace20e97826e9c0b0e966b7b1cd6f2aec6")
  flutter_litert_fetch_lib("${FLUTTER_LITERT_DXIL_DLL}"
    "${FLUTTER_LITERT_GPU_LIBS_URL}/dxil.dll"
    "cbcfe883a09fd0ca1f98abdf3a9553b560895e3283a136da82a8381253a169df")
  list(APPEND flutter_litert_bundled_libraries
    "${FLUTTER_LITERT_LITERT_DLL}"
    "${FLUTTER_LITERT_GPU_DLL}"
    "${FLUTTER_LITERT_DXCOMPILER_DLL}"
    "${FLUTTER_LITERT_DXIL_DLL}"
  )
else()
  message(STATUS "flutter_litert: FLUTTER_LITERT_BUNDLE_DESKTOP_RUNTIME is OFF; not bundling libLiteRt.dll, libLiteRtWebGpuAccelerator.dll, dxcompiler.dll or dxil.dll")
endif()

# Flutter's generated_plugins.cmake references plugins by their directory name
# (flutter_litert), not by PROJECT_NAME (flutter_tflite). Set both to ensure
# bundled native libraries are included.
set(flutter_tflite_bundled_libraries ${flutter_litert_bundled_libraries})

set(flutter_tflite_bundled_libraries ${flutter_tflite_bundled_libraries} PARENT_SCOPE)
set(flutter_litert_bundled_libraries ${flutter_litert_bundled_libraries} PARENT_SCOPE)

# === Tests ===
if (${include_${PROJECT_NAME}_tests})
set(TEST_RUNNER "${PROJECT_NAME}_test")
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of googletest" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(${TEST_RUNNER}
  test/flutter_tflite_plugin_test.cpp
  ${PLUGIN_SOURCES}
)
apply_standard_settings(${TEST_RUNNER})
target_include_directories(${TEST_RUNNER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TEST_RUNNER} PRIVATE flutter_wrapper_plugin)
target_link_libraries(${TEST_RUNNER} PRIVATE gtest_main gmock)
add_custom_command(TARGET ${TEST_RUNNER} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${FLUTTER_LIBRARY}" $<TARGET_FILE_DIR:${TEST_RUNNER}>
)

include(GoogleTest)
gtest_discover_tests(${TEST_RUNNER})
endif()
