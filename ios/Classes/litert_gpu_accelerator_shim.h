#ifndef FLUTTER_LITERT_LITERT_GPU_ACCELERATOR_SHIM_H_
#define FLUTTER_LITERT_LITERT_GPU_ACCELERATOR_SHIM_H_

// Keeps the LiteRtRegisterGpuAccelerator shim object linked into the app.
// The shim is only ever called by the LiteRT runtime through
// dlsym(RTLD_DEFAULT), which the linker cannot see, so without an anchor the
// object file would be dropped from the static archive (the CocoaPods pod
// target or the SwiftPM flutter_litert_gpu_shim target) at link time.
void FlutterLitertRetainLiteRtGpuShim(void);

#endif  // FLUTTER_LITERT_LITERT_GPU_ACCELERATOR_SHIM_H_
