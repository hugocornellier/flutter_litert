// Android runtime shim for flutter_litert.
//
// Every Android build ships this library. It answers which LiteRT runtime the
// app was built with (the flutterLitert.runtime Gradle property) so Dart can
// pick its native libraries synchronously, from any isolate.
//
// In the Play services mode it also owns the CompiledModel runtime handed out
// by Google Play services. Java resolves LiteRt.initializeNative() and passes
// the CompiledModelNativeHandle to nativeAttachCompiledModel(), which extracts
// the LiteRtRuntimeCApiStruct table with the play-services-litert prefab
// client. The library then exports one plain LiteRt* C function per symbol the
// Dart bindings use, each forwarding to the table (litert_forwarders.inc), so
// Dart binds this library exactly as it binds libLiteRt.so in the bundled mode.
//
// The Interpreter API needs no shim: TfLiteNative.initialize() initializes
// libtensorflowlite_jni_gms_client.so from Java, and Dart binds its Gms*
// exports directly. Java only reports the outcome here so helper isolates can
// see it.

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <jni.h>
#include <pthread.h>

#include "flutter_litert_runtime.h"

enum {
  kFlutterLiteRtRuntimeBundled = 0,
  kFlutterLiteRtRuntimePlayServices = 1,
};

// Readiness of one Play services runtime component. Values are shared with
// lib/src/bindings/android_runtime_shim.dart.
enum {
  kFlutterLiteRtStateNotInitialized = 0,
  kFlutterLiteRtStateReady = 1,
  kFlutterLiteRtStateFailed = 2,
};

static atomic_int g_interpreter_state = kFlutterLiteRtStateNotInitialized;
static atomic_int g_compiled_model_state = kFlutterLiteRtStateNotInitialized;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
#define FLUTTER_LITERT_ERROR_SIZE 512
static char g_interpreter_error[FLUTTER_LITERT_ERROR_SIZE];
static char g_compiled_model_error[FLUTTER_LITERT_ERROR_SIZE];

static void set_error(char* buffer, const char* message) {
  snprintf(buffer, FLUTTER_LITERT_ERROR_SIZE, "%s",
           message == NULL ? "" : message);
}

FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeMode(void) {
#if defined(FLUTTER_LITERT_PLAY_SERVICES)
  return kFlutterLiteRtRuntimePlayServices;
#else
  return kFlutterLiteRtRuntimeBundled;
#endif
}

FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeInterpreterState(void) {
  return atomic_load(&g_interpreter_state);
}

FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeCompiledModelState(void) {
  return atomic_load(&g_compiled_model_state);
}

// The returned strings are written before the matching state is published
// and never change afterwards, until a later retry rewrites them under
// g_lock before publishing a new state.
FLUTTER_LITERT_EXPORT const char* FlutterLiteRtRuntimeInterpreterError(void) {
  return g_interpreter_error;
}

FLUTTER_LITERT_EXPORT const char* FlutterLiteRtRuntimeCompiledModelError(void) {
  return g_compiled_model_error;
}

JNIEXPORT void JNICALL
Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeSetInterpreterState(
    JNIEnv* env, jclass clazz, jint state, jstring error) {
  (void)clazz;
  pthread_mutex_lock(&g_lock);
  if (atomic_load(&g_interpreter_state) != kFlutterLiteRtStateReady) {
    const char* message =
        error == NULL ? NULL : (*env)->GetStringUTFChars(env, error, NULL);
    set_error(g_interpreter_error, message);
    if (message != NULL) (*env)->ReleaseStringUTFChars(env, error, message);
    atomic_store(&g_interpreter_state, state);
  }
  pthread_mutex_unlock(&g_lock);
}

#if defined(FLUTTER_LITERT_PLAY_SERVICES)

#include <stddef.h>

#include "litert/abi/litert_c.h"
#include "litert/c/internal/litert_runtime_c_api.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

// Play services' RuntimeProxy treats every external table as ABI 1.0.0
// (b/574171444), which ends where litert_get_num_accelerators begins.
// Entries past that point are only safe once a runtime-version check exists.
#define FLUTTER_LITERT_ABI_1_0_END \
  offsetof(LiteRtRuntimeCApiStruct, litert_get_num_accelerators)

_Static_assert(FLUTTER_LITERT_ABI_1_0_END == 165 * sizeof(void*),
               "ABI 1.0.0 of LiteRtRuntimeCApiStruct has 165 entries");

// lib/src/bindings/litert_ffi.dart hard-codes these to pass the handles below
// as environment options.
_Static_assert(kLiteRtAnyTypeVoidPtr == 9, "LiteRtAnyType changed");
_Static_assert(kLiteRtEnvOptionTagContext == 28, "env option tags changed");
_Static_assert(kLiteRtEnvOptionTagSystemRuntimeHandle == 23,
               "env option tags changed");
_Static_assert(kLiteRtEnvOptionTagSystemGpuAcceleratorHandle == 30,
               "env option tags changed");

static const LiteRtRuntimeCApiStruct* _Atomic g_api = NULL;
static const void* g_gpu_accelerator = NULL;
static const void* g_context = NULL;
// The table is valid only while the Java handles stay alive, so they are
// pinned for the life of the process.
static jobject g_handle_ref = NULL;
static jobject g_internal_handle_ref = NULL;

FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeApi(void) {
  return atomic_load(&g_api);
}

FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeGpuAccelerator(void) {
  return atomic_load(&g_api) == NULL ? NULL : g_gpu_accelerator;
}

FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeContext(void) {
  return atomic_load(&g_api) == NULL ? NULL : g_context;
}

// Clears a pending Java exception and describes it in [buffer]. Returns
// whether an exception was pending.
static int take_exception(JNIEnv* env, const char* step, char* buffer,
                          size_t size) {
  jthrowable exception = (*env)->ExceptionOccurred(env);
  if (exception == NULL) return 0;
  (*env)->ExceptionClear(env);
  const char* description = NULL;
  jstring text = NULL;
  jclass throwable = (*env)->FindClass(env, "java/lang/Throwable");
  if (throwable != NULL) {
    jmethodID to_string =
        (*env)->GetMethodID(env, throwable, "toString", "()Ljava/lang/String;");
    if (to_string != NULL) {
      text = (jstring)(*env)->CallObjectMethod(env, exception, to_string);
    }
  }
  if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
  if (text != NULL) description = (*env)->GetStringUTFChars(env, text, NULL);
  snprintf(buffer, size, "%s threw %s", step,
           description == NULL ? "a Java exception" : description);
  if (description != NULL) (*env)->ReleaseStringUTFChars(env, text, description);
  return 1;
}

// Returns NULL on success, otherwise writes a description into [error].
static const char* attach(JNIEnv* env, jobject handle, char* error,
                          size_t size) {
  if (handle == NULL) return "CompiledModelNativeHandle is null";

  jobject internal = GmsLiteRtGetInternalHandle(env, handle);
  if (take_exception(env, "GmsLiteRtGetInternalHandle", error, size)) {
    return error;
  }
  if (internal == NULL) {
    return "GmsLiteRtGetInternalHandle returned no handle";
  }

  const char* failure = NULL;
  uintptr_t api = GmsLiteRtGetCompiledModelApi(env, internal);
  if (take_exception(env, "GmsLiteRtGetCompiledModelApi", error, size)) {
    failure = error;
  } else if (api == 0) {
    failure = "Google Play services returned no LiteRT runtime table";
  }

  uintptr_t gpu = 0;
  if (failure == NULL) {
    gpu = GmsLiteRtGetGpuAccelerator(env, internal);
    if (take_exception(env, "GmsLiteRtGetGpuAccelerator", error, size)) {
      failure = error;
    }
  }
  if (failure == NULL) {
    GmsLiteRtInitialize(env, internal);
    if (take_exception(env, "GmsLiteRtInitialize", error, size)) {
      failure = error;
    }
  }
  if (failure == NULL) {
    g_handle_ref = (*env)->NewGlobalRef(env, handle);
    g_internal_handle_ref = (*env)->NewGlobalRef(env, internal);
    if (g_handle_ref == NULL || g_internal_handle_ref == NULL) {
      failure = "could not pin the Play services runtime handle";
    }
  }
  (*env)->DeleteLocalRef(env, internal);
  if (failure != NULL) {
    if (g_handle_ref != NULL) (*env)->DeleteGlobalRef(env, g_handle_ref);
    if (g_internal_handle_ref != NULL) {
      (*env)->DeleteGlobalRef(env, g_internal_handle_ref);
    }
    g_handle_ref = NULL;
    g_internal_handle_ref = NULL;
    return failure;
  }

  g_gpu_accelerator = (const void*)gpu;
  g_context = GmsLiteRtGetContext();
  // Publishing the table last makes the pointers above visible to any thread
  // that observes a non-null table.
  atomic_store(&g_api, (const LiteRtRuntimeCApiStruct*)api);
  return NULL;
}

// Returns null once the runtime is attached (now or by an earlier call), or a
// description of the failure. A failed attach can be retried.
JNIEXPORT jstring JNICALL
Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeAttachCompiledModel(
    JNIEnv* env, jclass clazz, jobject handle) {
  (void)clazz;
  char error[FLUTTER_LITERT_ERROR_SIZE];
  const char* failure = NULL;
  pthread_mutex_lock(&g_lock);
  if (atomic_load(&g_api) == NULL) {
    failure = attach(env, handle, error, sizeof(error));
    set_error(g_compiled_model_error, failure);
    atomic_store(&g_compiled_model_state, failure == NULL
                                              ? kFlutterLiteRtStateReady
                                              : kFlutterLiteRtStateFailed);
  }
  pthread_mutex_unlock(&g_lock);
  return failure == NULL ? NULL : (*env)->NewStringUTF(env, failure);
}

JNIEXPORT void JNICALL
Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeSetCompiledModelFailure(
    JNIEnv* env, jclass clazz, jstring error) {
  (void)clazz;
  pthread_mutex_lock(&g_lock);
  if (atomic_load(&g_api) == NULL) {
    const char* message =
        error == NULL ? NULL : (*env)->GetStringUTFChars(env, error, NULL);
    set_error(g_compiled_model_error, message);
    if (message != NULL) (*env)->ReleaseStringUTFChars(env, error, message);
    atomic_store(&g_compiled_model_state, kFlutterLiteRtStateFailed);
  }
  pthread_mutex_unlock(&g_lock);
}

// Dart only binds these after FlutterLiteRtRuntimeCompiledModelState()
// reports ready, so a null table here is a plugin bug rather than a user
// error; fail the call instead of crashing.
#define FLUTTER_LITERT_CHECK_SLOT(name, field)                         \
  _Static_assert(offsetof(LiteRtRuntimeCApiStruct, field) <            \
                     FLUTTER_LITERT_ABI_1_0_END,                       \
                 #name " is outside the LiteRT runtime ABI 1.0.0 table")

#define FLUTTER_LITERT_FORWARD_STATUS(name, field, params, args) \
  FLUTTER_LITERT_CHECK_SLOT(name, field);                        \
  FLUTTER_LITERT_EXPORT LiteRtStatus name params {               \
    const LiteRtRuntimeCApiStruct* api = atomic_load(&g_api);    \
    if (api == NULL || api->field == NULL) {                     \
      return kLiteRtStatusErrorRuntimeFailure;                   \
    }                                                            \
    return api->field args;                                      \
  }

#define FLUTTER_LITERT_FORWARD_VOID(name, field, params, args) \
  FLUTTER_LITERT_CHECK_SLOT(name, field);                      \
  FLUTTER_LITERT_EXPORT void name params {                     \
    const LiteRtRuntimeCApiStruct* api = atomic_load(&g_api);  \
    if (api == NULL || api->field == NULL) return;             \
    api->field args;                                           \
  }

#include "litert_forwarders.inc"

#else  // !FLUTTER_LITERT_PLAY_SERVICES

FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeApi(void) { return NULL; }

FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeGpuAccelerator(void) {
  return NULL;
}

FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeContext(void) {
  return NULL;
}

#endif  // FLUTTER_LITERT_PLAY_SERVICES
