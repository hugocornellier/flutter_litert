#ifndef FLUTTER_LITERT_COREML_NPU_ACCELERATOR_H_
#define FLUTTER_LITERT_COREML_NPU_ACCELERATOR_H_

// Keeps the iOS Core ML NPU registration bridge linked into the app. Dart
// resolves its exported entry points through dlsym(RTLD_DEFAULT), which the
// native linker cannot otherwise see.
void FlutterLitertRetainLiteRtCoreMlNpuShim(void);

#endif  // FLUTTER_LITERT_COREML_NPU_ACCELERATOR_H_
