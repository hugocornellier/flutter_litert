// Exports of libflutter_litert_runtime.so (see flutter_litert_runtime.c).
// lib/src/bindings/android_runtime_shim.dart binds these by name.
#ifndef FLUTTER_LITERT_RUNTIME_H_
#define FLUTTER_LITERT_RUNTIME_H_

#include <jni.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLUTTER_LITERT_EXPORT __attribute__((visibility("default")))

// The flutterLitert.runtime the app was built with. 0: bundled runtime,
// 1: Google Play services runtime, 2: auto (both packaged).
FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeMode(void);

// The runtime in use for the whole process: 0 or 1, or -1 while an auto build
// has not chosen yet. Outside the auto mode this is the build's runtime.
FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeSelected(void);

// Chooses 0 or 1 for an auto build, unless a runtime was already chosen, and
// returns the runtime in use. The first choice holds for the process, and
// play_services_error, which may be null, is kept with it.
// Outside the auto mode this changes nothing.
FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeSelect(
    int32_t runtime, const char* play_services_error);

// Why an auto build chose the bundled runtime. Empty otherwise.
FLUTTER_LITERT_EXPORT const char* FlutterLiteRtRuntimePlayServicesError(void);

// 0: not initialized, 1: ready, 2: initialization failed.
FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeInterpreterState(void);
FLUTTER_LITERT_EXPORT int32_t FlutterLiteRtRuntimeCompiledModelState(void);

// Empty unless the matching state is 2.
FLUTTER_LITERT_EXPORT const char* FlutterLiteRtRuntimeInterpreterError(void);
FLUTTER_LITERT_EXPORT const char* FlutterLiteRtRuntimeCompiledModelError(void);

// Why the ready CompiledModel runtime would have no GPU accelerator, recorded
// by the call that initialized it. Empty until then.
FLUTTER_LITERT_EXPORT const char* FlutterLiteRtRuntimeGpuError(void);

// Play services values for LiteRtEnvOption tags 23, 30 and 28. All null until
// the CompiledModel runtime is ready, and always null in the bundled mode.
FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeApi(void);
FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeGpuAccelerator(void);
FLUTTER_LITERT_EXPORT const void* FlutterLiteRtRuntimeContext(void);

// Called by com.hugocornellier.flutter_litert.PlayServicesRuntime.
JNIEXPORT void JNICALL
Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeSetInterpreterState(
    JNIEnv* env, jclass clazz, jint state, jstring error);
JNIEXPORT jstring JNICALL
Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeAttachCompiledModel(
    JNIEnv* env, jclass clazz, jobject handle, jstring gpu_error);
JNIEXPORT void JNICALL
Java_com_hugocornellier_flutter_1litert_PlayServicesRuntime_nativeSetCompiledModelFailure(
    JNIEnv* env, jclass clazz, jstring error);

#ifdef __cplusplus
}
#endif

#endif  // FLUTTER_LITERT_RUNTIME_H_
