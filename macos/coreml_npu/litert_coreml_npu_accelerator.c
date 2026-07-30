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

// macOS-only LiteRT accelerator registration bridge.
//
// LiteRT's built-in NPU accelerator is a vendor-dispatch backend. Apple does
// not provide a LiteRT dispatch implementation for the Neural Engine, so this
// bridge registers the TensorFlow Lite CoreML delegate as a LiteRT accelerator
// advertising kLiteRtHwAcceleratorNpu.
//
// The sibling libtensorflowlite_coreml_npu-mac.dylib is a dedicated build of
// the upstream CoreML delegate whose MLModelConfiguration is restricted to
// MLComputeUnitsCPUAndNeuralEngine. It is intentionally separate from the
// classic Interpreter delegate, which retains its upstream MLComputeUnitsAll
// behavior.
//
// This file mirrors only the stable C signatures needed to register an
// accelerator. It resolves them dynamically from the already-loaded
// libLiteRt.dylib so the bridge never links a second copy of the runtime.

#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <TargetConditionals.h>

#if !TARGET_OS_OSX
#error "The CoreML NPU accelerator bridge is macOS-only."
#endif

#if !defined(__arm64__)
#error "The CoreML NPU accelerator bridge requires Apple Silicon."
#endif

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

// Mirrors TfLiteCoreMlDelegateOptions without depending on TensorFlow Lite
// headers in this small bridge.
typedef struct {
  int enabled_devices;
  int coreml_version;
  int max_delegated_partitions;
  int min_nodes_per_partition;
} TfLiteCoreMlDelegateOptionsShim;

enum {
  // The upstream device-name probe recognizes iPhones and iPads only. macOS
  // capability is enforced by the arm64 build and the dedicated delegate's
  // MLComputeUnitsCPUAndNeuralEngine configuration instead.
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
typedef void (*LiteRtDestroyDelegateCallback)(
    LiteRtRuntimeContext* runtime_context, LiteRtDelegateWrapper delegate);
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
    LiteRtCreateDelegateCallback create_delegate,
    LiteRtDestroyDelegateCallback destroy_delegate);
typedef LiteRtStatus (*LiteRtWrapDelegateFn)(
    void* tflite_delegate, LiteRtDelegateWrapper* wrapper);
typedef LiteRtStatus (*LiteRtUnwrapDelegateFn)(
    LiteRtDelegateWrapper wrapper, void** tflite_delegate);

// Prefix of LiteRtAcceleratorDefV1 through the delegate callbacks. The shipped
// LiteRT runtime exports its static XNNPACK definition; registering that
// definition after Core ML preserves NPU-first ordering while still giving
// mixed {npu, cpu} models XNNPACK for any remaining graph.
typedef struct {
  int version;
  LiteRtGetNameCallback get_name;
  LiteRtGetVersionCallback get_version;
  LiteRtGetHardwareSupportCallback get_hardware_support;
  LiteRtIsJitCallback is_jit;
  LiteRtCreateDelegateCallback create_delegate;
  LiteRtDestroyDelegateCallback destroy_delegate;
} LiteRtAcceleratorDefV1Prefix;

// Keep the two hand-mirrored ABI prefixes honest. LiteRT v2.1.5 publishes the
// accelerator offsets below as its version-1 ABI, and Core ML's public options
// contain four consecutive 32-bit values in non-debug builds.
_Static_assert(sizeof(TfLiteCoreMlDelegateOptionsShim) == 16,
               "TfLite Core ML options ABI mismatch");
_Static_assert(offsetof(LiteRtAcceleratorDefV1Prefix, get_name) == 8,
               "LiteRt accelerator get_name offset mismatch");
_Static_assert(offsetof(LiteRtAcceleratorDefV1Prefix, create_delegate) == 40,
               "LiteRt accelerator create_delegate offset mismatch");
_Static_assert(offsetof(LiteRtAcceleratorDefV1Prefix, destroy_delegate) == 48,
               "LiteRt accelerator destroy_delegate offset mismatch");
_Static_assert(sizeof(LiteRtAcceleratorDefV1Prefix) == 56,
               "LiteRt accelerator prefix size mismatch");

typedef void* (*TfLiteCoreMlDelegateCreateFn)(
    const TfLiteCoreMlDelegateOptionsShim* options);
typedef void (*TfLiteCoreMlDelegateDeleteFn)(void* delegate);
typedef int (*TfLiteCoreMlDelegateGetLastDelegatedNodeCountFn)(void);

typedef struct {
  void* handle;
  LiteRtCreateAcceleratorFn create_accelerator;
  LiteRtDestroyAcceleratorFn destroy_accelerator;
  LiteRtRegisterAcceleratorFn register_accelerator;
  LiteRtSetAcceleratorGetNameFn set_get_name;
  LiteRtSetAcceleratorGetVersionFn set_get_version;
  LiteRtSetAcceleratorGetHardwareSupportFn set_get_hardware_support;
  LiteRtSetIsJitFn set_is_jit;
  LiteRtSetDelegateFunctionFn set_delegate_function;
  LiteRtWrapDelegateFn wrap_delegate;
  LiteRtUnwrapDelegateFn unwrap_delegate;
  const LiteRtAcceleratorDefV1Prefix* cpu_accelerator_def;
  LiteRtStatus load_status;
} LiteRtSymbols;

typedef struct {
  void* handle;
  TfLiteCoreMlDelegateCreateFn create;
  TfLiteCoreMlDelegateDeleteFn destroy;
  TfLiteCoreMlDelegateGetLastDelegatedNodeCountFn
      get_last_delegated_node_count;
  LiteRtStatus load_status;
} CoreMlSymbols;

static LiteRtSymbols g_litert;
static CoreMlSymbols g_coreml;
static pthread_once_t g_litert_once = PTHREAD_ONCE_INIT;
static pthread_once_t g_coreml_once = PTHREAD_ONCE_INIT;

// Forward declaration used to find this dylib's directory with dladdr().
__attribute__((visibility("default"))) LiteRtStatus
FlutterLiteRtRegisterCoreMlNpuAccelerator(LiteRtEnvironment environment);

static void LogDynamicLoadError(const char* library, const char* detail) {
  fprintf(stderr, "[flutter_litert] CoreML NPU: failed to load %s: %s\n",
          library, detail != NULL ? detail : "unknown dynamic-loader error");
}

static void* OpenSiblingLibrary(const char* library_name) {
  Dl_info self_info;
  if (dladdr((const void*)&FlutterLiteRtRegisterCoreMlNpuAccelerator,
             &self_info) == 0 ||
      self_info.dli_fname == NULL) {
    LogDynamicLoadError(library_name, "could not resolve bridge location");
    return NULL;
  }

  char path[PATH_MAX];
  const size_t self_length = strlen(self_info.dli_fname);
  if (self_length >= sizeof(path)) {
    LogDynamicLoadError(library_name, "bridge path exceeds PATH_MAX");
    return NULL;
  }
  memcpy(path, self_info.dli_fname, self_length + 1);

  char* separator = strrchr(path, '/');
  if (separator == NULL) {
    LogDynamicLoadError(library_name, "bridge path has no parent directory");
    return NULL;
  }
  separator[1] = '\0';

  const size_t parent_length = strlen(path);
  const size_t library_length = strlen(library_name);
  if (parent_length + library_length >= sizeof(path)) {
    LogDynamicLoadError(library_name, "sibling path exceeds PATH_MAX");
    return NULL;
  }
  memcpy(path + parent_length, library_name, library_length + 1);

  void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    LogDynamicLoadError(path, dlerror());
  }
  return handle;
}

static void* ResolveRequired(void* handle, const char* symbol) {
  dlerror();
  void* value = dlsym(handle, symbol);
  const char* error = dlerror();
  if (value == NULL || error != NULL) {
    LogDynamicLoadError(symbol, error);
    return NULL;
  }
  return value;
}

static void LoadLiteRtSymbols(void) {
  memset(&g_litert, 0, sizeof(g_litert));
  g_litert.load_status = kLiteRtStatusErrorRuntimeFailure;
  g_litert.handle = OpenSiblingLibrary("libLiteRt.dylib");
  if (g_litert.handle == NULL) {
    return;
  }

#define LOAD_LITERT(field, symbol)                                      \
  do {                                                                  \
    g_litert.field = (void*)ResolveRequired(g_litert.handle, (symbol)); \
    if (g_litert.field == NULL) {                                       \
      return;                                                           \
    }                                                                   \
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
  LOAD_LITERT(unwrap_delegate, "LiteRtUnwrapDelegate");

#undef LOAD_LITERT

  const LiteRtAcceleratorDefV1Prefix* const* cpu_def =
      (const LiteRtAcceleratorDefV1Prefix* const*)ResolveRequired(
          g_litert.handle, "LiteRtStaticLinkedAcceleratorCpuDef");
  if (cpu_def == NULL || *cpu_def == NULL || (*cpu_def)->version != 1) {
    return;
  }
  g_litert.cpu_accelerator_def = *cpu_def;
  g_litert.load_status = kLiteRtStatusOk;
}

static void LoadCoreMlSymbols(void) {
  memset(&g_coreml, 0, sizeof(g_coreml));
  g_coreml.load_status = kLiteRtStatusErrorUnsupported;

  const char* override_path =
      getenv("FLUTTER_LITERT_COREML_NPU_DELEGATE_PATH");
  if (override_path != NULL && override_path[0] != '\0') {
    g_coreml.handle = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
    if (g_coreml.handle == NULL) {
      LogDynamicLoadError(override_path, dlerror());
      return;
    }
  } else {
    g_coreml.handle =
        OpenSiblingLibrary("libtensorflowlite_coreml_npu-mac.dylib");
    if (g_coreml.handle == NULL) {
      return;
    }
  }

  g_coreml.create = (TfLiteCoreMlDelegateCreateFn)ResolveRequired(
      g_coreml.handle, "TfLiteCoreMlDelegateCreate");
  g_coreml.destroy = (TfLiteCoreMlDelegateDeleteFn)ResolveRequired(
      g_coreml.handle, "TfLiteCoreMlDelegateDelete");
  g_coreml.get_last_delegated_node_count =
      (TfLiteCoreMlDelegateGetLastDelegatedNodeCountFn)ResolveRequired(
          g_coreml.handle,
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
  // The CoreML delegate converts and compiles each delegated partition while
  // the LiteRT compiled model applies it.
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
  if (g_litert.load_status != kLiteRtStatusOk) {
    return g_litert.load_status;
  }
  pthread_once(&g_coreml_once, LoadCoreMlSymbols);
  if (g_coreml.load_status != kLiteRtStatusOk) {
    return g_coreml.load_status;
  }

  const TfLiteCoreMlDelegateOptionsShim coreml_options = {
      .enabled_devices = kTfLiteCoreMlDelegateAllDevices,
      .coreml_version = 3,
      .max_delegated_partitions = 0,
      .min_nodes_per_partition = 2,
  };
  void* delegate = g_coreml.create(&coreml_options);
  if (delegate == NULL) {
    return kLiteRtStatusErrorUnsupported;
  }

  const LiteRtStatus wrap_status =
      g_litert.wrap_delegate(delegate, delegate_wrapper);
  if (wrap_status != kLiteRtStatusOk) {
    g_coreml.destroy(delegate);
  }
  return wrap_status;
}

static void DestroyDelegate(LiteRtRuntimeContext* runtime_context,
                            LiteRtDelegateWrapper delegate_wrapper) {
  (void)runtime_context;
  if (delegate_wrapper == NULL) {
    return;
  }
  pthread_once(&g_litert_once, LoadLiteRtSymbols);
  pthread_once(&g_coreml_once, LoadCoreMlSymbols);
  if (g_litert.load_status != kLiteRtStatusOk ||
      g_coreml.load_status != kLiteRtStatusOk) {
    return;
  }

  void* delegate = NULL;
  if (g_litert.unwrap_delegate(delegate_wrapper, &delegate) ==
          kLiteRtStatusOk &&
      delegate != NULL) {
    g_coreml.destroy(delegate);
  }
}

static LiteRtStatus RegisterCpuFallback(LiteRtEnvironment environment) {
  const LiteRtAcceleratorDefV1Prefix* def = g_litert.cpu_accelerator_def;
  if (def == NULL || def->get_name == NULL || def->get_version == NULL ||
      def->get_hardware_support == NULL || def->is_jit == NULL ||
      def->create_delegate == NULL || def->destroy_delegate == NULL) {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtAccelerator accelerator = NULL;
  LiteRtStatus status = g_litert.create_accelerator(&accelerator);
  if (status != kLiteRtStatusOk) {
    return status;
  }

#define SET_CPU_OR_DESTROY(call)                  \
  do {                                            \
    status = (call);                              \
    if (status != kLiteRtStatusOk) {              \
      g_litert.destroy_accelerator(accelerator);  \
      return status;                              \
    }                                             \
  } while (0)

  SET_CPU_OR_DESTROY(g_litert.set_get_name(accelerator, def->get_name));
  SET_CPU_OR_DESTROY(
      g_litert.set_get_version(accelerator, def->get_version));
  SET_CPU_OR_DESTROY(g_litert.set_get_hardware_support(
      accelerator, def->get_hardware_support));
  SET_CPU_OR_DESTROY(g_litert.set_is_jit(accelerator, def->is_jit));
  SET_CPU_OR_DESTROY(g_litert.set_delegate_function(
      accelerator, def->create_delegate, def->destroy_delegate));

#undef SET_CPU_OR_DESTROY

  return g_litert.register_accelerator(environment, accelerator, NULL, NULL);
}

LiteRtStatus FlutterLiteRtRegisterCoreMlNpuAccelerator(
    LiteRtEnvironment environment) {
  if (environment == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  pthread_once(&g_litert_once, LoadLiteRtSymbols);
  if (g_litert.load_status != kLiteRtStatusOk) {
    return g_litert.load_status;
  }
  pthread_once(&g_coreml_once, LoadCoreMlSymbols);
  if (g_coreml.load_status != kLiteRtStatusOk) {
    return g_coreml.load_status;
  }

  LiteRtAccelerator accelerator = NULL;
  LiteRtStatus status = g_litert.create_accelerator(&accelerator);
  if (status != kLiteRtStatusOk) {
    return status;
  }

#define SET_OR_DESTROY(call)                         \
  do {                                               \
    status = (call);                                 \
    if (status != kLiteRtStatusOk) {                 \
      g_litert.destroy_accelerator(accelerator);     \
      return status;                                 \
    }                                                \
  } while (0)

  SET_OR_DESTROY(g_litert.set_get_name(accelerator, GetName));
  SET_OR_DESTROY(g_litert.set_get_version(accelerator, GetVersion));
  SET_OR_DESTROY(
      g_litert.set_get_hardware_support(accelerator, GetHardwareSupport));
  SET_OR_DESTROY(
      g_litert.set_is_jit(accelerator, IsDelegateResponsibleForJit));
  SET_OR_DESTROY(g_litert.set_delegate_function(
      accelerator, CreateDelegate, DestroyDelegate));

#undef SET_OR_DESTROY

  // LiteRtRegisterAccelerator assumes ownership even when registration fails.
  status =
      g_litert.register_accelerator(environment, accelerator, NULL, NULL);
  if (status != kLiteRtStatusOk) {
    return status;
  }

  // CPU is registered only after Core ML so it acts as the fallback rather
  // than claiming the graph first. If the runtime ever omits its XNNPACK
  // definition, TFLite's built-in CPU kernels still provide correct fallback.
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
