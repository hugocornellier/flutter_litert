cmake_minimum_required(VERSION 3.10)
project(tflite_custom_ops C)

set(CMAKE_C_STANDARD 11)

# Source files for custom ops
set(CUSTOM_OPS_SOURCES
    custom_ops/transpose_conv_bias.c
)

# Create shared library for custom ops
add_library(tflite_custom_ops SHARED ${CUSTOM_OPS_SOURCES})

# Include directories - set to src/ for "tensorflow_lite/common.h" style includes
target_include_directories(tflite_custom_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Export all symbols (needed for FFI lookup)
set_target_properties(tflite_custom_ops PROPERTIES
    C_VISIBILITY_PRESET default
)

# Platform-specific settings
if(ANDROID)
    set_target_properties(tflite_custom_ops PROPERTIES
        OUTPUT_NAME "tflite_custom_ops"
    )

    # Ensure the library is built with proper symbol visibility
    target_compile_options(tflite_custom_ops PRIVATE
        -fvisibility=default
    )
elseif(WIN32)
    set_target_properties(tflite_custom_ops PROPERTIES
        OUTPUT_NAME "tflite_custom_ops"
        SUFFIX ".dll"
    )
endif()
