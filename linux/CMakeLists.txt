cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "flutter_tflite")
project(${PROJECT_NAME} LANGUAGES CXX C)

set(PLUGIN_NAME "flutter_tflite_plugin")

list(APPEND PLUGIN_SOURCES
  "flutter_tflite_plugin.cc"
)

add_library(${PLUGIN_NAME} SHARED
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
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter)
target_link_libraries(${PLUGIN_NAME} PRIVATE PkgConfig::GTK)

# Build the TFLite custom ops library (e.g., Convolution2DTransposeBias for MediaPipe)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/custom_ops")

# The WebGPU GPU accelerator (~18 MB) is fetched at build time from a GitHub
# release rather than bundled in the published pub.dev package, which must stay
# under the 100 MiB limit. Downloaded (verified by SHA-256) only if it is not
# already present with the pinned hash.
# It is the unmodified file from the same ai-edge-litert wheel as libLiteRt.so
# and must match it: the runtime/accelerator ABI changed between 2.1.5 and 2.3.0.
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
  foreach(attempt RANGE 1 3)
    message(STATUS "flutter_litert: downloading ${url} (attempt ${attempt})")
    file(DOWNLOAD "${url}" "${path}"
      EXPECTED_HASH SHA256=${sha}
      TLS_VERIFY ON
      STATUS dl_status)
    list(GET dl_status 0 dl_code)
    if(dl_code EQUAL 0)
      return()
    endif()
    list(GET dl_status 1 dl_msg)
    message(WARNING "flutter_litert: download failed (${dl_msg}); retrying")
    file(REMOVE "${path}")
  endforeach()
  message(FATAL_ERROR "flutter_litert: could not download ${url}")
endfunction()

flutter_litert_fetch_lib(
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRtWebGpuAccelerator.so"
  "https://github.com/hugocornellier/flutter_litert/releases/download/litert-desktop-gpu-v2.3.0/libLiteRtWebGpuAccelerator.so"
  "b67f193ae6eb8ad3a15349f54aba696c97d6f981c92bc2eb1096cbe9e2835b25")

set(flutter_tflite_bundled_libraries
  "$<TARGET_FILE:tflite_custom_ops>"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtensorflowlite_c-linux.so"
  # LiteRT Next runtime (CompiledModel API) + WebGPU/Vulkan GPU accelerator,
  # x86_64; from the official ai-edge-litert 2.3.0 PyPI wheel. The runtime
  # discovers the accelerator by scanning its own directory for the bare
  # soname, so both must land in the same bundled lib/ dir.
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRt.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRtWebGpuAccelerator.so"
)

# Flutter's generated_plugins.cmake references plugins by their directory name
# (flutter_litert), not by PROJECT_NAME (flutter_tflite). Set both to ensure
# the custom ops library gets bundled.
set(flutter_litert_bundled_libraries
  "$<TARGET_FILE:tflite_custom_ops>"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtensorflowlite_c-linux.so"
  # LiteRT Next runtime (CompiledModel API) + WebGPU/Vulkan GPU accelerator,
  # x86_64; from the official ai-edge-litert 2.3.0 PyPI wheel. The runtime
  # discovers the accelerator by scanning its own directory for the bare
  # soname, so both must land in the same bundled lib/ dir.
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRt.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLiteRtWebGpuAccelerator.so"
)

set(flutter_tflite_bundled_libraries ${flutter_tflite_bundled_libraries} PARENT_SCOPE)
set(flutter_litert_bundled_libraries ${flutter_litert_bundled_libraries} PARENT_SCOPE)

# === Tests ===
if (${include_${PROJECT_NAME}_tests})
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
message("Unit tests require CMake 3.11.0 or later")
else()
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
  test/flutter_tflite_plugin_test.cc
  ${PLUGIN_SOURCES}
)
apply_standard_settings(${TEST_RUNNER})
target_include_directories(${TEST_RUNNER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TEST_RUNNER} PRIVATE flutter)
target_link_libraries(${TEST_RUNNER} PRIVATE PkgConfig::GTK)
target_link_libraries(${TEST_RUNNER} PRIVATE gtest_main gmock)

include(GoogleTest)
gtest_discover_tests(${TEST_RUNNER})

endif()  # CMake version check
endif()  # include_${PROJECT_NAME}_tests
