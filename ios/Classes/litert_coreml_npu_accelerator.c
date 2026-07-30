/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// iOS LiteRT accelerator registration bridge.
//
// Apple does not ship a LiteRT dispatch backend for the Neural Engine. This
// bridge registers flutter_litert's patched TensorFlow Lite Core ML delegate
// as an accelerator advertising kLiteRtHwAcceleratorNpu. The patched delegate
// has a dedicated entry point that selects MLComputeUnitsCPUAndNeuralEngine;
// its ordinary TfLiteCoreMlDelegateCreate entry point remains unchanged.
//
// LiteRT and TensorFlowLiteCCoreML are framework-wrapped in iOS apps, so their
// C APIs are resolved from the already-loaded process image. This avoids
// linking another LiteRT runtime and works for both CocoaPods and SwiftPM.

#include <TargetConditionals.h>

#include "litert_coreml_npu_accelerator.h"

#if TARGET_OS_IPHONE

#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int32_t LiteRtStatus;
typedef struct LiteRtAcceleratorT* LiteRtAccelerator;
typedef struct LiteRtEnvironmentT* LiteRtEnvironment;
typedef struct LiteRtOptionsT* LiteRtOptions;
typedef struct LiteRtDelegateWrapperT* LiteRtDelegateWrapper;
typedef struct LiteRtRuntimeContext LiteRtRuntimeContext;

typedef struct {
  int major;
  int minor;
  int patch;
} LiteRtApiVersion;

enum {
  kLiteRtStatusOk = 0,
  kLiteRtStatusErrorInvalidArgument = 1,
  kLiteRtStatusErrorRuntimeFailure = 3,
  kLiteRtStatusErrorUnsupported = 5,
  kLiteRtHwAcceleratorNpu = 4,
};

// Mirrors TfLiteCoreMlDelegateOptions without importing TensorFlow Lite
// headers into this ABI bridge.
typedef struct {
  int enabled_devices;
  int coreml_version;
  int max_delegated_partitions;
  int min_nodes_per_partition;
} TfLiteCoreMlDelegateOptionsShim;

enum {
  kTfLiteCoreMlDelegateDevicesWithNeuralEngine = 0,
  kTfLiteCoreMlDelegateAllDevices = 1,
};

typedef LiteRtStatus (*LiteRtCreateAcceleratorFn)(
    LiteRtAccelerator* accelerator);
typedef LiteRtStatus (*LiteRtDestroyAcceleratorFn)(
    LiteRtAccelerator accelerator);
typedef LiteRtStatus (*LiteRtRegisterAcceleratorFn)(
    LiteRtEnvironment environment, LiteRtAccelerator accelerator, void* data,
    void (*release_data)(void*));
typedef LiteRtStatus (*LiteRtGetNameCallback)(
    LiteRtAccelerator accelerator, const char** name);
typedef LiteRtStatus (*LiteRtGetVersionCallback)(
    LiteRtAccelerator accelerator, LiteRtApiVersion* version);
typedef LiteRtStatus (*LiteRtGetHardwareSupportCallback)(
    LiteRtAccelerator accelerator, int* supported_hardware);
typedef LiteRtStatus (*LiteRtIsJitCallback)(
    LiteRtAccelerator accelerator, bool* does_jit_compilation);
typedef LiteRtStatus (*LiteRtCreateDelegateCallback)(
    LiteRtRuntimeContext* runtime_context, LiteRtEnvironment environment,
    LiteRtAccelerator accelerator, LiteRtOptions options,
    LiteRtDelegateWrapper* delegate);
typedef LiteRtStatus (*LiteRtSetAcceleratorGetNameFn)(
    LiteRtAccelerator accelerator, LiteRtGetNameCallback get_name);
typedef LiteRtStatus (*LiteRtSetAcceleratorGetVersionFn)(
    LiteRtAccelerator accelerator, LiteRtGetVersionCallback get_version);
typedef LiteRtStatus (*LiteRtSetAcceleratorGetHardwareSupportFn)(
    LiteRtAccelerator accelerator,
    LiteRtGetHardwareSupportCallback get_hardware_support);
typedef LiteRtStatus (*LiteRtSetIsJitFn)(
    LiteRtAccelerator accelerator, LiteRtIsJitCallback is_jit);
typedef LiteRtStatus (*LiteRtSetDelegateFunctionFn)(
    LiteRtAccelerator accelerator,
    LiteRtCreateDelegateCallback create_delegate);
typedef void (*TfLiteDelegateDeleter)(void* tflite_delegate);
typedef LiteRtStatus (*LiteRtWrapDelegateFn)(
    void* tflite_delegate, TfLiteDelegateDeleter deleter,
    LiteRtDelegateWrapper* wrapper);

// The iOS LiteRT binaries are pinned to commit
// 1adc2475829fbe52d5670873821a45bea8779532. At that revision delegate wrappers
// own the delegate through a deleter passed to LiteRtWrapDelegate; there is no
// separate destroy-delegate callback in LiteRtAcceleratorDefV1. Registering
// the runtime's XNNPACK definition after Core ML preserves NPU-first ordering
// for a mixed {npu, cpu} request.
typedef struct {
  int version;
  LiteRtGetNameCallback get_name;
  LiteRtGetVersionCallback get_version;
  LiteRtGetHardwareSupportCallback get_hardware_support;
  LiteRtIsJitCallback is_jit;
  LiteRtCreateDelegateCallback create_delegate;
} LiteRtAcceleratorDefV1Prefix;

_Static_assert(sizeof(TfLiteCoreMlDelegateOptionsShim) == 16,
               "TfLite Core ML options ABI mismatch");
_Static_assert(offsetof(LiteRtAcceleratorDefV1Prefix, get_name) == 8,
               "LiteRt accelerator get_name offset mismatch");
_Static_assert(offsetof(LiteRtAcceleratorDefV1Prefix, create_delegate) == 40,
               "LiteRt accelerator create_delegate offset mismatch");
_Static_assert(sizeof(LiteRtAcceleratorDefV1Prefix) == 48,
               "LiteRt accelerator prefix size mismatch");

typedef void* (*TfLiteCoreMlNpuDelegateCreateFn)(
    const TfLiteCoreMlDelegateOptionsShim* options);
typedef void (*TfLiteCoreMlDelegateDeleteFn)(void* delegate);
typedef int (*TfLiteCoreMlDelegateGetLastDelegatedNodeCountFn)(void);

typedef struct {
  LiteRtCreateAcceleratorFn create_accelerator;
  LiteRtDestroyAcceleratorFn destroy_accelerator;
  LiteRtRegisterAcceleratorFn register_accelerator;
  LiteRtSetAcceleratorGetNameFn set_get_name;
  LiteRtSetAcceleratorGetVersionFn set_get_version;
  LiteRtSetAcceleratorGetHardwareSupportFn set_get_hardware_support;
  LiteRtSetIsJitFn set_is_jit;
  LiteRtSetDelegateFunctionFn set_delegate_function;
  LiteRtWrapDelegateFn wrap_delegate;
  const LiteRtAcceleratorDefV1Prefix* cpu_accelerator_def;
  LiteRtStatus load_status;
} LiteRtSymbols;

typedef struct {
  TfLiteCoreMlNpuDelegateCreateFn create;
  TfLiteCoreMlDelegateDeleteFn destroy;
  TfLiteCoreMlDelegateGetLastDelegatedNodeCountFn
      get_last_delegated_node_count;
  LiteRtStatus load_status;
} CoreMlSymbols;

static LiteRtSymbols g_litert;
static CoreMlSymbols g_coreml;
static pthread_once_t g_litert_once = PTHREAD_ONCE_INIT;
static pthread_once_t g_coreml_once = PTHREAD_ONCE_INIT;

static void LogSymbolError(const char* symbol) {
  const char* detail = dlerror();
  fprintf(stderr, "[flutter_litert] CoreML NPU: missing symbol %s: %s\n",
          symbol, detail != NULL ? detail : "not exported by the app");
}

static void* ResolveRequired(const char* symbol) {
  dlerror();
  void* value = dlsym(RTLD_DEFAULT, symbol);
  const char* error = dlerror();
  if (value == NULL || error != NULL) {
    LogSymbolError(symbol);
    return NULL;
  }
  return value;
}

static void LoadLiteRtSymbols(void) {
  memset(&g_litert, 0, sizeof(g_litert));
  g_litert.load_status = kLiteRtStatusErrorRuntimeFailure;

#define LOAD_LITERT(field, symbol)                               \
  do {                                                            \
    g_litert.field = (void*)ResolveRequired((symbol));            \
    if (g_litert.field == NULL) {                                 \
      return;                                                     \
    }                                                             \
  } while (0)

  LOAD_LITERT(create_accelerator, "LiteRtCreateAccelerator");
  LOAD_LITERT(destroy_accelerator, "LiteRtDestroyAccelerator");
  LOAD_LITERT(register_accelerator, "LiteRtRegisterAccelerator");
  LOAD_LITERT(set_get_name, "LiteRtSetAcceleratorGetName");
  LOAD_LITERT(set_get_version, "LiteRtSetAcceleratorGetVersion");
  LOAD_LITERT(set_get_hardware_support,
              "LiteRtSetAcceleratorGetHardwareSupport");
  LOAD_LITERT(
      set_is_jit,
      "LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation");
  LOAD_LITERT(set_delegate_function, "LiteRtSetDelegateFunction");
  LOAD_LITERT(wrap_delegate, "LiteRtWrapDelegate");

#undef LOAD_LITERT

  const LiteRtAcceleratorDefV1Prefix* const* cpu_def =
      (const LiteRtAcceleratorDefV1Prefix* const*)ResolveRequired(
          "LiteRtStaticLinkedAcceleratorCpuDef");
  if (cpu_def == NULL || *cpu_def == NULL || (*cpu_def)->version != 1) {
    return;
  }
  g_litert.cpu_accelerator_def = *cpu_def;
  g_litert.load_status = kLiteRtStatusOk;
}

static void LoadCoreMlSymbols(void) {
  memset(&g_coreml, 0, sizeof(g_coreml));
  g_coreml.load_status = kLiteRtStatusErrorUnsupported;
  g_coreml.create = (TfLiteCoreMlNpuDelegateCreateFn)ResolveRequired(
      "FlutterTfLiteCoreMlNpuDelegateCreate");
  g_coreml.destroy = (TfLiteCoreMlDelegateDeleteFn)ResolveRequired(
      "TfLiteCoreMlDelegateDelete");
  g_coreml.get_last_delegated_node_count =
      (TfLiteCoreMlDelegateGetLastDelegatedNodeCountFn)ResolveRequired(
          "FlutterTfLiteCoreMlNpuGetLastDelegatedNodeCount");
  if (g_coreml.create == NULL || g_coreml.destroy == NULL ||
      g_coreml.get_last_delegated_node_count == NULL) {
    return;
  }
  g_coreml.load_status = kLiteRtStatusOk;
}

static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                            const char** name) {
  (void)accelerator;
  if (name == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *name = "CoreML Neural Engine";
  return kLiteRtStatusOk;
}

static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                               LiteRtApiVersion* version) {
  (void)accelerator;
  if (version == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *version = (LiteRtApiVersion){1, 0, 0};
  return kLiteRtStatusOk;
}

static LiteRtStatus GetHardwareSupport(LiteRtAccelerator accelerator,
                                       int* supported_hardware) {
  (void)accelerator;
  if (supported_hardware == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

static LiteRtStatus IsDelegateResponsibleForJit(
    LiteRtAccelerator accelerator, bool* does_jit_compilation) {
  (void)accelerator;
  if (does_jit_compilation == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *does_jit_compilation = true;
  return kLiteRtStatusOk;
}

static LiteRtStatus CreateDelegate(
    LiteRtRuntimeContext* runtime_context, LiteRtEnvironment environment,
    LiteRtAccelerator accelerator, LiteRtOptions options,
    LiteRtDelegateWrapper* delegate_wrapper) {
  (void)runtime_context;
  (void)environment;
  (void)accelerator;
  (void)options;
  if (delegate_wrapper == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  pthread_once(&g_litert_once, LoadLiteRtSymbols);
  pthread_once(&g_coreml_once, LoadCoreMlSymbols);
  if (g_litert.load_status != kLiteRtStatusOk) {
    return g_litert.load_status;
  }
  if (g_coreml.load_status != kLiteRtStatusOk) {
    return g_coreml.load_status;
  }

  const TfLiteCoreMlDelegateOptionsShim coreml_options = {
#if TARGET_OS_SIMULATOR
      // A simulator has no ANE device identifier. Core ML still accepts the
      // CPUAndNeuralEngine policy and executes the delegated model on CPU,
      // which validates the integration without claiming hardware execution.
      .enabled_devices = kTfLiteCoreMlDelegateAllDevices,
#else
      .enabled_devices = kTfLiteCoreMlDelegateDevicesWithNeuralEngine,
#endif
      .coreml_version = 3,
      .max_delegated_partitions = 0,
      .min_nodes_per_partition = 2,
  };
  void* delegate = g_coreml.create(&coreml_options);
  if (delegate == NULL) {
    return kLiteRtStatusErrorUnsupported;
  }

  const LiteRtStatus wrap_status =
      g_litert.wrap_delegate(delegate, g_coreml.destroy, delegate_wrapper);
  if (wrap_status != kLiteRtStatusOk) {
    g_coreml.destroy(delegate);
  }
  return wrap_status;
}

static LiteRtStatus RegisterCpuFallback(LiteRtEnvironment environment) {
  const LiteRtAcceleratorDefV1Prefix* def = g_litert.cpu_accelerator_def;
  if (def == NULL || def->get_name == NULL || def->get_version == NULL ||
      def->get_hardware_support == NULL || def->is_jit == NULL ||
      def->create_delegate == NULL) {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtAccelerator accelerator = NULL;
  LiteRtStatus status = g_litert.create_accelerator(&accelerator);
  if (status != kLiteRtStatusOk) {
    return status;
  }

#define SET_CPU_OR_DESTROY(call)                   \
  do {                                             \
    status = (call);                               \
    if (status != kLiteRtStatusOk) {               \
      g_litert.destroy_accelerator(accelerator);   \
      return status;                               \
    }                                              \
  } while (0)

  SET_CPU_OR_DESTROY(g_litert.set_get_name(accelerator, def->get_name));
  SET_CPU_OR_DESTROY(
      g_litert.set_get_version(accelerator, def->get_version));
  SET_CPU_OR_DESTROY(g_litert.set_get_hardware_support(
      accelerator, def->get_hardware_support));
  SET_CPU_OR_DESTROY(g_litert.set_is_jit(accelerator, def->is_jit));
  SET_CPU_OR_DESTROY(
      g_litert.set_delegate_function(accelerator, def->create_delegate));

#undef SET_CPU_OR_DESTROY

  return g_litert.register_accelerator(
      environment, accelerator, NULL, NULL);
}

__attribute__((visibility("default"))) LiteRtStatus
FlutterLiteRtRegisterCoreMlNpuAccelerator(LiteRtEnvironment environment) {
  if (environment == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  pthread_once(&g_litert_once, LoadLiteRtSymbols);
  pthread_once(&g_coreml_once, LoadCoreMlSymbols);
  if (g_litert.load_status != kLiteRtStatusOk) {
    return g_litert.load_status;
  }
  if (g_coreml.load_status != kLiteRtStatusOk) {
    return g_coreml.load_status;
  }

  LiteRtAccelerator accelerator = NULL;
  LiteRtStatus status = g_litert.create_accelerator(&accelerator);
  if (status != kLiteRtStatusOk) {
    return status;
  }

#define SET_OR_DESTROY(call)                       \
  do {                                             \
    status = (call);                               \
    if (status != kLiteRtStatusOk) {               \
      g_litert.destroy_accelerator(accelerator);   \
      return status;                               \
    }                                              \
  } while (0)

  SET_OR_DESTROY(g_litert.set_get_name(accelerator, GetName));
  SET_OR_DESTROY(g_litert.set_get_version(accelerator, GetVersion));
  SET_OR_DESTROY(
      g_litert.set_get_hardware_support(accelerator, GetHardwareSupport));
  SET_OR_DESTROY(
      g_litert.set_is_jit(accelerator, IsDelegateResponsibleForJit));
  SET_OR_DESTROY(
      g_litert.set_delegate_function(accelerator, CreateDelegate));

#undef SET_OR_DESTROY

  // LiteRtRegisterAccelerator assumes ownership even when registration fails.
  status =
      g_litert.register_accelerator(environment, accelerator, NULL, NULL);
  if (status != kLiteRtStatusOk) {
    return status;
  }

  const LiteRtStatus cpu_status = RegisterCpuFallback(environment);
  if (cpu_status != kLiteRtStatusOk) {
    fprintf(stderr,
            "[flutter_litert] CoreML NPU: XNNPACK fallback registration "
            "failed (LiteRtStatus=%d); using built-in CPU kernels.\n",
            cpu_status);
  }
  return kLiteRtStatusOk;
}

__attribute__((visibility("default"))) int
FlutterLiteRtCoreMlNpuGetLastDelegatedNodeCount(void) {
  pthread_once(&g_coreml_once, LoadCoreMlSymbols);
  if (g_coreml.load_status != kLiteRtStatusOk ||
      g_coreml.get_last_delegated_node_count == NULL) {
    return -1;
  }
  return g_coreml.get_last_delegated_node_count();
}

__attribute__((visibility("default"))) void
FlutterLitertRetainLiteRtCoreMlNpuShim(void) {
  volatile void* keep =
      (void*)&FlutterLiteRtRegisterCoreMlNpuAccelerator;
  keep = (void*)&FlutterLiteRtCoreMlNpuGetLastDelegatedNodeCount;
  (void)keep;
}

#else  // !TARGET_OS_IPHONE

void FlutterLitertRetainLiteRtCoreMlNpuShim(void) {}

#endif  // TARGET_OS_IPHONE
