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
    
    target_link_options(tflite_custom_ops PRIVATE
        -Wl,-z,max-page-size=16384
    )
elseif(WIN32)
    set_target_properties(tflite_custom_ops PROPERTIES
        OUTPUT_NAME "tflite_custom_ops"
        SUFFIX ".dll"
    )
endif()

# Android runtime shim (play_services/flutter_litert_runtime.c). It reports
# the runtime selected by the flutterLitert.runtime Gradle property and, for
# the Google Play services runtime, forwards the LiteRt C API to the function
# table that Play services provides.
if(ANDROID)
    set(FLUTTER_LITERT_RUNTIME "bundled" CACHE STRING
        "LiteRT runtime: bundled, playServices or auto")

    add_library(flutter_litert_runtime SHARED
        play_services/flutter_litert_runtime.c
    )
    set_target_properties(flutter_litert_runtime PROPERTIES
        C_VISIBILITY_PRESET hidden
    )
    # missing-prototypes makes the compiler check every generated forwarder
    # against the public LiteRt declaration of the same name.
    target_compile_options(flutter_litert_runtime PRIVATE
        -Wall -Wextra -Werror=missing-prototypes
    )
    target_link_options(flutter_litert_runtime PRIVATE
        -Wl,-z,max-page-size=16384
    )

    # auto links the Play services client too, and also packages the bundled
    # runtime; initializeAndroidRuntime() chooses one of them at run time.
    if(FLUTTER_LITERT_RUNTIME STREQUAL "playServices" OR
       FLUTTER_LITERT_RUNTIME STREQUAL "auto")
        find_package(litert_compiledmodel_gms_client REQUIRED CONFIG)
        target_compile_definitions(flutter_litert_runtime PRIVATE
            FLUTTER_LITERT_PLAY_SERVICES=1
        )
        if(FLUTTER_LITERT_RUNTIME STREQUAL "auto")
            target_compile_definitions(flutter_litert_runtime PRIVATE
                FLUTTER_LITERT_RUNTIME_AUTO=1
            )
        endif()
        # The generated build_config.h the prefab headers include must come
        # before the prefab include directory.
        target_include_directories(flutter_litert_runtime BEFORE PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/play_services/include
        )
        target_link_libraries(flutter_litert_runtime PRIVATE
            litert_compiledmodel_gms_client::litert_compiledmodel_gms_client
        )
    elseif(NOT FLUTTER_LITERT_RUNTIME STREQUAL "bundled")
        message(FATAL_ERROR
            "FLUTTER_LITERT_RUNTIME must be bundled, playServices or auto, "
            "not '${FLUTTER_LITERT_RUNTIME}'")
    endif()
endif()
