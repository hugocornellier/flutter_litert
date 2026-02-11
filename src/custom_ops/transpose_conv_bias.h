// Copyright 2018 The TensorFlow Authors.
// Copyright 2019 The MediaPipe Authors.
// Copyright 2025 flutter_litert authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Standalone implementation of MediaPipe's Convolution2DTransposeBias custom op
// for use with flutter_litert. Uses only the public TFLite C API.

#ifndef TFLITE_FLUTTER_CUSTOM_TRANSPOSE_CONV_BIAS_H_
#define TFLITE_FLUTTER_CUSTOM_TRANSPOSE_CONV_BIAS_H_

// Platform-specific TFLite header includes
#if (defined(__APPLE__) && TARGET_OS_IOS) || defined(TFLITE_USE_FRAMEWORK_HEADERS)
// iOS: Use framework headers from CocoaPods
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
// Desktop: Use headers from local path (set via CMake/header search paths)
#include "tensorflow_lite/common.h"
#include "tensorflow_lite/c_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Symbol export for dynamic lookup via FFI
// 'used' prevents linker from stripping the symbol even if unreferenced
// 'visibility' makes it visible for dynamic lookup
#if defined(_WIN32)
#define TFLITE_CUSTOM_OPS_EXPORT __declspec(dllexport)
#else
#define TFLITE_CUSTOM_OPS_EXPORT __attribute__((used, visibility("default")))
#endif

// Returns the TfLiteRegistration for the Convolution2DTransposeBias custom op.
// This must be registered with the interpreter options before creating
// an interpreter that uses models containing this custom op.
//
// Usage from Dart:
//   final registration = tfliteBinding.TfLiteFlutter_RegisterConvolution2DTransposeBias();
//   tfliteBinding.TfLiteInterpreterOptionsAddCustomOp(
//     options, "Convolution2DTransposeBias".toNativeUtf8(), registration, 1, 1);
TFLITE_CUSTOM_OPS_EXPORT TfLiteRegistration* TfLiteFlutter_RegisterConvolution2DTransposeBias(void);

#ifdef __cplusplus
}
#endif

#endif  // TFLITE_FLUTTER_CUSTOM_TRANSPOSE_CONV_BIAS_H_
