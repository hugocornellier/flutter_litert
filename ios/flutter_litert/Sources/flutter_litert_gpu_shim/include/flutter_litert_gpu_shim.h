#ifndef FLUTTER_LITERT_GPU_SHIM_H_
#define FLUTTER_LITERT_GPU_SHIM_H_

// Keeps the LiteRtRegisterGpuAccelerator shim object linked into the
// flutter-litert dynamic product. The shim is only ever called by the LiteRT
// runtime through dlsym(RTLD_DEFAULT), which the linker cannot see, so
// without an anchor the object file would be dropped from the target's
// static archive when the product links.
void FlutterLitertRetainLiteRtGpuShim(void);

#endif  // FLUTTER_LITERT_GPU_SHIM_H_
