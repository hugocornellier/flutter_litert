cmake_minimum_required(VERSION 3.10)
project(tflite_custom_ops C)

set(CMAKE_C_STANDARD 11)

# Source files
set(SOURCES
    transpose_conv_bias.c
)

# Create shared library
add_library(tflite_custom_ops SHARED ${SOURCES})

# Include directories - set to src/ so we can use "tensorflow_lite/common.h" style includes
target_include_directories(tflite_custom_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Platform-specific settings
if(APPLE)
    # Link against the TFLite dylib
    set(TFLITE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../macos/libtensorflowlite_c-mac.dylib")
    target_link_libraries(tflite_custom_ops PRIVATE ${TFLITE_LIB})

    set_target_properties(tflite_custom_ops PROPERTIES
        OUTPUT_NAME "tflite_custom_ops"
        SUFFIX ".dylib"
        # Don't embed the path to TFLite dylib - it will be found at runtime
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(WIN32)
    set_target_properties(tflite_custom_ops PROPERTIES
        OUTPUT_NAME "tflite_custom_ops"
        SUFFIX ".dll"
    )
else()
    set_target_properties(tflite_custom_ops PROPERTIES
        OUTPUT_NAME "tflite_custom_ops"
        SUFFIX ".so"
    )
endif()

# Export symbols
set_target_properties(tflite_custom_ops PROPERTIES
    C_VISIBILITY_PRESET default
)
