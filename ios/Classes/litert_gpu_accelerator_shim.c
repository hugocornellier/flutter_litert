// GPU (Metal) accelerator registration shim for iOS.
//
// Both distribution channels ship the LiteRT Next runtime and Metal
// accelerator as conventional .framework bundles: CocoaPods refuses to
// install vendored xcframeworks that contain bare dynamic libraries ("Use
// dynamic frameworks for dynamic linking instead"), and App Store validation
// rejects IPAs whose Frameworks/ directory holds the loose dylibs Xcode
// embeds for bare-dylib SwiftPM binary targets (ITMS-90426, issue #15). The
// framework shape breaks LiteRT's own GPU plugin discovery, which dlopen's
// the exact file name `libLiteRtMetalAccelerator.dylib` from the
// RuntimeLibraryDir environment option and reads its exported
// `LiteRtAcceleratorImpl` definition.
//
// LiteRT's registration has a documented second probe: it looks up a
// function named `LiteRtRegisterGpuAccelerator` and, if the dlopen by file
// name fails, retries the lookup against the whole process (RTLD_DEFAULT).
// This file exports that function. When the runtime calls it, the shim
// dlopen's the framework-wrapped Metal accelerator from the app bundle's
// Frameworks directory and registers its accelerator definition through the
// exported LiteRT C ABI — replicating upstream
// `litert::internal::RegisterAcceleratorFromDef`.
//
// ABI: `LiteRtAcceleratorDefV1` is declared ABI-stable by LiteRT
// (litert/c/internal/litert_accelerator_def.h, version field +
// static_asserts). The layout below matches google-ai-edge/LiteRT commit
// 1adc2475829fbe52d5670873821a45bea8779532 — the same commit the bundled iOS
// binaries are built from — and is guarded by the same static_asserts plus a
// runtime version check.
//
// CocoaPods compiles this file via the podspec's Classes/** source glob;
// SwiftPM compiles it through the Sources/flutter_litert_gpu_shim forwarder
// target. (SwiftPM builds of package versions before the #15 fix shipped
// bare dylibs, where LiteRT's own file-name scan succeeded first and this
// function was never called.)

#include <TargetConditionals.h>

#include "litert_gpu_accelerator_shim.h"

#if TARGET_OS_IPHONE

#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// ---- Minimal mirrors of the LiteRT C ABI (pinned, see header comment) ----

typedef int32_t LiteRtStatusShim;  // LiteRtStatus is a C enum (int).
enum {
  kShimLiteRtStatusOk = 0,
  kShimLiteRtStatusErrorInvalidArgument = 1,
  kShimLiteRtStatusErrorRuntimeFailure = 3,
  kShimLiteRtStatusErrorNotFound = 6,
  kShimLiteRtStatusErrorWrongVersion = 8,
};

#define SHIM_ACCELERATOR_DEF_VERSION 1
#define SHIM_MAX_SUPPORTED_BUFFER_TYPES 16

// All LiteRT handle and callback types are opaque pointers at this ABI
// boundary; only the struct layout below must match.
typedef struct {
  int version;
  void* get_name;
  void* get_version;
  void* get_hardware_support;
  void* is_tflite_delegate_responsible_for_jit_compilation;
  void* create_delegate;
  void* start_metrics_collection;
  void* stop_metrics_collection;
  struct {
    void* create_func;
    void* destroy_func;
    void* lock_func;
    void* unlock_func;
    void* clear_func;
    void* import_func;
    int32_t device_tag;  // LiteRtEnvOptionTag (C enum).
    int32_t queue_tag;   // LiteRtEnvOptionTag (C enum).
    size_t num_supported_buffer_types;
    int32_t supported_buffer_types[SHIM_MAX_SUPPORTED_BUFFER_TYPES];
  } buffer_handlers;
} ShimLiteRtAcceleratorDefV1;

// Mirrors of upstream litert_accelerator_def.h /
// litert_custom_tensor_buffer_handlers_def.h static_asserts.
_Static_assert(sizeof(ShimLiteRtAcceleratorDefV1) == 192,
               "LiteRtAcceleratorDefV1 size mismatch");
_Static_assert(offsetof(ShimLiteRtAcceleratorDefV1, get_name) == 8,
               "get_name offset mismatch");
_Static_assert(offsetof(ShimLiteRtAcceleratorDefV1, create_delegate) == 40,
               "create_delegate offset mismatch");
_Static_assert(offsetof(ShimLiteRtAcceleratorDefV1, buffer_handlers) == 64,
               "buffer_handlers offset mismatch");
_Static_assert(
    offsetof(ShimLiteRtAcceleratorDefV1, buffer_handlers.device_tag) == 112,
    "device_tag offset mismatch");  // 64 + 48 per upstream asserts.
_Static_assert(offsetof(ShimLiteRtAcceleratorDefV1,
                        buffer_handlers.num_supported_buffer_types) == 120,
               "num_supported_buffer_types offset mismatch");  // 64 + 56.
_Static_assert(offsetof(ShimLiteRtAcceleratorDefV1,
                        buffer_handlers.supported_buffer_types) == 128,
               "supported_buffer_types offset mismatch");  // 64 + 64.

typedef LiteRtStatusShim (*ShimCreateAccelerator)(void** accelerator);
typedef LiteRtStatusShim (*ShimDestroyAccelerator)(void* accelerator);
typedef LiteRtStatusShim (*ShimSetFnPtr)(void* accelerator, void* fn);
typedef LiteRtStatusShim (*ShimRegisterAccelerator)(void* env,
                                                    void* accelerator,
                                                    void* data,
                                                    void (*release)(void*));
typedef LiteRtStatusShim (*ShimRegisterTensorBufferHandlers)(
    void* env, int32_t buffer_type, void* create_func, void* destroy_func,
    void* lock_func, void* unlock_func, void* clear_func, void* import_func,
    int32_t device_tag, int32_t queue_tag);

static void* shim_api(const char* name) { return dlsym(RTLD_DEFAULT, name); }

// Returns the dlopen'd handle for the framework-wrapped Metal accelerator,
// or NULL. The handle is intentionally never dlclose'd: the runtime keeps
// registered accelerators alive for the process lifetime.
static void* shim_open_metal_accelerator(void) {
  CFBundleRef bundle = CFBundleGetMainBundle();
  if (bundle == NULL) return NULL;
  CFURLRef frameworks_url = CFBundleCopyPrivateFrameworksURL(bundle);
  if (frameworks_url == NULL) return NULL;

  char frameworks_path[1024];
  bool ok = CFURLGetFileSystemRepresentation(
      frameworks_url, true, (UInt8*)frameworks_path, sizeof(frameworks_path));
  CFRelease(frameworks_url);
  if (!ok) return NULL;

  char path[1280];
  int written = snprintf(
      path, sizeof(path),
      "%s/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator",
      frameworks_path);
  if (written <= 0 || (size_t)written >= sizeof(path)) return NULL;
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

// Probe target for LiteRT's GPU accelerator auto-registration. See the file
// header for why this exists. Must keep this exact name.
LiteRtStatusShim LiteRtRegisterGpuAccelerator(void* env) {
  void* handle = shim_open_metal_accelerator();
  if (handle == NULL) return kShimLiteRtStatusErrorNotFound;

  const ShimLiteRtAcceleratorDefV1* def =
      (const ShimLiteRtAcceleratorDefV1*)dlsym(handle, "LiteRtAcceleratorImpl");
  if (def == NULL) return kShimLiteRtStatusErrorNotFound;
  if (def->version != SHIM_ACCELERATOR_DEF_VERSION) {
    return kShimLiteRtStatusErrorWrongVersion;
  }
  if (def->get_name == NULL || def->get_version == NULL ||
      def->get_hardware_support == NULL ||
      def->is_tflite_delegate_responsible_for_jit_compilation == NULL ||
      def->create_delegate == NULL ||
      def->buffer_handlers.num_supported_buffer_types >=
          SHIM_MAX_SUPPORTED_BUFFER_TYPES) {
    return kShimLiteRtStatusErrorInvalidArgument;
  }

  ShimCreateAccelerator create_accelerator =
      (ShimCreateAccelerator)shim_api("LiteRtCreateAccelerator");
  ShimDestroyAccelerator destroy_accelerator =
      (ShimDestroyAccelerator)shim_api("LiteRtDestroyAccelerator");
  ShimSetFnPtr set_get_name =
      (ShimSetFnPtr)shim_api("LiteRtSetAcceleratorGetName");
  ShimSetFnPtr set_get_version =
      (ShimSetFnPtr)shim_api("LiteRtSetAcceleratorGetVersion");
  ShimSetFnPtr set_get_hardware_support =
      (ShimSetFnPtr)shim_api("LiteRtSetAcceleratorGetHardwareSupport");
  ShimSetFnPtr set_delegate_function =
      (ShimSetFnPtr)shim_api("LiteRtSetDelegateFunction");
  ShimSetFnPtr set_jit_responsibility = (ShimSetFnPtr)shim_api(
      "LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation");
  ShimRegisterAccelerator register_accelerator =
      (ShimRegisterAccelerator)shim_api("LiteRtRegisterAccelerator");
  ShimRegisterTensorBufferHandlers register_buffer_handlers =
      (ShimRegisterTensorBufferHandlers)shim_api(
          "LiteRtRegisterTensorBufferHandlers");
  if (create_accelerator == NULL || destroy_accelerator == NULL ||
      set_get_name == NULL || set_get_version == NULL ||
      set_get_hardware_support == NULL || set_delegate_function == NULL ||
      set_jit_responsibility == NULL || register_accelerator == NULL ||
      register_buffer_handlers == NULL) {
    return kShimLiteRtStatusErrorRuntimeFailure;
  }

  void* accelerator = NULL;
  LiteRtStatusShim status = create_accelerator(&accelerator);
  if (status != kShimLiteRtStatusOk) return status;

#define SHIM_SET_OR_BAIL(call)        \
  do {                                \
    status = (call);                  \
    if (status != kShimLiteRtStatusOk) { \
      destroy_accelerator(accelerator);  \
      return status;                  \
    }                                 \
  } while (0)

  SHIM_SET_OR_BAIL(set_get_name(accelerator, def->get_name));
  SHIM_SET_OR_BAIL(set_get_version(accelerator, def->get_version));
  SHIM_SET_OR_BAIL(
      set_get_hardware_support(accelerator, def->get_hardware_support));
  SHIM_SET_OR_BAIL(set_delegate_function(accelerator, def->create_delegate));
  SHIM_SET_OR_BAIL(set_jit_responsibility(
      accelerator,
      def->is_tflite_delegate_responsible_for_jit_compilation));
#undef SHIM_SET_OR_BAIL

  // On failure LiteRtRegisterAccelerator releases the accelerator itself.
  status = register_accelerator(env, accelerator, NULL, NULL);
  if (status != kShimLiteRtStatusOk) return status;

  for (size_t i = 0; i < def->buffer_handlers.num_supported_buffer_types;
       ++i) {
    status = register_buffer_handlers(
        env, def->buffer_handlers.supported_buffer_types[i],
        def->buffer_handlers.create_func, def->buffer_handlers.destroy_func,
        def->buffer_handlers.lock_func, def->buffer_handlers.unlock_func,
        def->buffer_handlers.clear_func, def->buffer_handlers.import_func,
        def->buffer_handlers.device_tag, def->buffer_handlers.queue_tag);
    if (status != kShimLiteRtStatusOk) return status;
  }

  return kShimLiteRtStatusOk;
}

void FlutterLitertRetainLiteRtGpuShim(void) {
  // Taking the address from a separately-linked caller keeps this object
  // file (and the exported LiteRtRegisterGpuAccelerator) in the app binary.
  volatile void* keep = (void*)&LiteRtRegisterGpuAccelerator;
  (void)keep;
}

#else  // !TARGET_OS_IPHONE

void FlutterLitertRetainLiteRtGpuShim(void) {}

#endif  // TARGET_OS_IPHONE
