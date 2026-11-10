// Smoke test for a built TensorFlow Lite C shared library.
//
// Usage: tflite_c_smoke <libtensorflowlite_c> <add.tflite> <expected-version-prefix>
//
// Loads the library with dlopen, prints TfLiteVersion(), checks it starts with
// the expected prefix, then runs test/assets/add.tflite (float32, x + x + x)
// on the CPU and through the XNNPACK delegate with every input set to 1.0, and
// checks every output is 3.0. Exits non-zero on any failure.
//
// Used by .github/workflows/build-tflite-c-macos.yml. The CMake build has two
// traps that produce a library reporting the wrong version (it can silently
// build TF 2.19.0 sources, or keep TF_MINOR_VERSION=19), so the version check
// is the load-bearing part.

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteTensor TfLiteTensor;
typedef struct TfLiteDelegate TfLiteDelegate;
typedef struct TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptions;

#define LOAD(name)                                     \
  name##_t name = (name##_t)dlsym(lib, #name);         \
  if (!name) {                                         \
    fprintf(stderr, "FAIL: missing symbol %s\n", #name); \
    return 1;                                          \
  }

typedef const char* (*TfLiteVersion_t)(void);
typedef TfLiteModel* (*TfLiteModelCreateFromFile_t)(const char*);
typedef TfLiteInterpreterOptions* (*TfLiteInterpreterOptionsCreate_t)(void);
typedef void (*TfLiteInterpreterOptionsSetNumThreads_t)(
    TfLiteInterpreterOptions*, int);
typedef void (*TfLiteInterpreterOptionsAddDelegate_t)(TfLiteInterpreterOptions*,
                                                       TfLiteDelegate*);
typedef TfLiteInterpreter* (*TfLiteInterpreterCreate_t)(
    const TfLiteModel*, const TfLiteInterpreterOptions*);
typedef int (*TfLiteInterpreterAllocateTensors_t)(TfLiteInterpreter*);
typedef int (*TfLiteInterpreterInvoke_t)(TfLiteInterpreter*);
typedef int (*TfLiteInterpreterGetInputTensorCount_t)(const TfLiteInterpreter*);
typedef int (*TfLiteInterpreterGetOutputTensorCount_t)(const TfLiteInterpreter*);
typedef TfLiteTensor* (*TfLiteInterpreterGetInputTensor_t)(
    const TfLiteInterpreter*, int);
typedef const TfLiteTensor* (*TfLiteInterpreterGetOutputTensor_t)(
    const TfLiteInterpreter*, int);
typedef size_t (*TfLiteTensorByteSize_t)(const TfLiteTensor*);
typedef void* (*TfLiteTensorData_t)(const TfLiteTensor*);
typedef int (*TfLiteTensorType_t)(const TfLiteTensor*);
typedef void (*TfLiteInterpreterDelete_t)(TfLiteInterpreter*);
typedef void (*TfLiteInterpreterOptionsDelete_t)(TfLiteInterpreterOptions*);
typedef void (*TfLiteModelDelete_t)(TfLiteModel*);
typedef TfLiteDelegate* (*TfLiteXNNPackDelegateCreate_t)(
    const TfLiteXNNPackDelegateOptions*);
typedef void (*TfLiteXNNPackDelegateDelete_t)(TfLiteDelegate*);

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <lib> <add.tflite> <version-prefix>\n", argv[0]);
    return 2;
  }
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "FAIL: dlopen: %s\n", dlerror());
    return 1;
  }
  LOAD(TfLiteVersion);
  LOAD(TfLiteModelCreateFromFile);
  LOAD(TfLiteInterpreterOptionsCreate);
  LOAD(TfLiteInterpreterOptionsSetNumThreads);
  LOAD(TfLiteInterpreterOptionsAddDelegate);
  LOAD(TfLiteInterpreterCreate);
  LOAD(TfLiteInterpreterAllocateTensors);
  LOAD(TfLiteInterpreterInvoke);
  LOAD(TfLiteInterpreterGetInputTensorCount);
  LOAD(TfLiteInterpreterGetOutputTensorCount);
  LOAD(TfLiteInterpreterGetInputTensor);
  LOAD(TfLiteInterpreterGetOutputTensor);
  LOAD(TfLiteTensorByteSize);
  LOAD(TfLiteTensorData);
  LOAD(TfLiteTensorType);
  LOAD(TfLiteInterpreterDelete);
  LOAD(TfLiteInterpreterOptionsDelete);
  LOAD(TfLiteModelDelete);
  LOAD(TfLiteXNNPackDelegateCreate);
  LOAD(TfLiteXNNPackDelegateDelete);

  const char* version = TfLiteVersion();
  printf("TfLiteVersion: %s\n", version);
  if (strncmp(version, argv[3], strlen(argv[3])) != 0) {
    fprintf(stderr, "FAIL: version %s does not start with %s\n", version,
            argv[3]);
    return 1;
  }

  TfLiteModel* model = TfLiteModelCreateFromFile(argv[2]);
  if (!model) {
    fprintf(stderr, "FAIL: model load\n");
    return 1;
  }

  for (int use_xnnpack = 0; use_xnnpack <= 1; ++use_xnnpack) {
    TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(opts, 2);
    TfLiteDelegate* xnn = NULL;
    if (use_xnnpack) {
      xnn = TfLiteXNNPackDelegateCreate(NULL);
      if (!xnn) {
        fprintf(stderr, "FAIL: XNNPACK delegate create\n");
        return 1;
      }
      TfLiteInterpreterOptionsAddDelegate(opts, xnn);
    }
    TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, opts);
    if (!interp || TfLiteInterpreterAllocateTensors(interp) != 0) {
      fprintf(stderr, "FAIL: interpreter create/allocate (xnnpack=%d)\n",
              use_xnnpack);
      return 1;
    }
    int n_in = TfLiteInterpreterGetInputTensorCount(interp);
    for (int i = 0; i < n_in; ++i) {
      TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp, i);
      if (TfLiteTensorType(t) != 1 /* kTfLiteFloat32 */) {
        fprintf(stderr, "FAIL: input %d is not float32\n", i);
        return 1;
      }
      float* d = (float*)TfLiteTensorData(t);
      size_t n = TfLiteTensorByteSize(t) / sizeof(float);
      for (size_t k = 0; k < n; ++k) d[k] = 1.0f;
    }
    if (TfLiteInterpreterInvoke(interp) != 0) {
      fprintf(stderr, "FAIL: invoke (xnnpack=%d)\n", use_xnnpack);
      return 1;
    }
    int n_out = TfLiteInterpreterGetOutputTensorCount(interp);
    for (int i = 0; i < n_out; ++i) {
      const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp, i);
      const float* d = (const float*)TfLiteTensorData(t);
      size_t n = TfLiteTensorByteSize(t) / sizeof(float);
      for (size_t k = 0; k < n; ++k) {
        if (fabsf(d[k] - 3.0f) > 1e-5f) {
          fprintf(stderr, "FAIL: output[%d][%zu] = %f, want 3.0 (xnnpack=%d)\n",
                  i, k, d[k], use_xnnpack);
          return 1;
        }
      }
      printf("ok: xnnpack=%d output %d has %zu values, all 3.0\n", use_xnnpack,
             i, n);
    }
    TfLiteInterpreterDelete(interp);
    TfLiteInterpreterOptionsDelete(opts);
    if (xnn) TfLiteXNNPackDelegateDelete(xnn);
  }
  TfLiteModelDelete(model);
  printf("PASS\n");
  return 0;
}
