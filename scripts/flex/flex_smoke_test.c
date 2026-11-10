// Smoke test for a desktop Flex delegate library (macOS dylib), used by
// .github/workflows/build-flex-macos.yml.
//
// Loads a TFLite C library and the Flex delegate library with dlopen (no
// headers or link-time dependencies), creates the delegate through the
// external delegate plugin entry point, and runs one step of the "train"
// signature of test/assets/training_model_flex.tflite. Inputs are
// zero-filled. That model's Flex ops (FlexSaveV2, FlexRestoreV2) are in its
// save and restore signatures, not in train, so this proves the delegate
// loads, is created and is destroyed, not that a Flex kernel runs.
//
// Usage: flex_smoke_test <libtensorflowlite_c> <flex dylib> <model> [--no-flex]
// Exit:  0 when the train step ran (or, with --no-flex, whatever happened:
//        that run is a control and only informative), 1 otherwise.
//
// Ends with _exit() so TensorFlow's atexit handlers do not run; they can
// crash at process exit (seen as SIGBUS in the Dart flex tests).

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteSignatureRunner TfLiteSignatureRunner;
typedef struct TfLiteTensor TfLiteTensor;
typedef struct TfLiteDelegate TfLiteDelegate;
typedef int TfLiteStatus;  // kTfLiteOk == 0
typedef int TfLiteType;    // kTfLiteFloat32 == 1

#define SYM(lib, type, name) \
  __typeof__(type) name = (__typeof__(type))dlsym(lib, #name); \
  if (!name) { fprintf(stderr, "missing symbol %s\n", #name); finish(1); }

static void finish(int code) {
  fflush(stdout);
  fflush(stderr);
  _exit(code);
}

static void reporter(void* user, const char* fmt, va_list args) {
  (void)user;
  fprintf(stderr, "[tflite] ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <tflite_c> <flex> <model> [--no-flex]\n", argv[0]);
    return 2;
  }
  int use_flex = !(argc > 4 && strcmp(argv[4], "--no-flex") == 0);

  void* c = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!c) { fprintf(stderr, "dlopen tflite_c: %s\n", dlerror()); finish(1); }

  SYM(c, const char* (*)(void), TfLiteVersion);
  SYM(c, TfLiteModel* (*)(const char*), TfLiteModelCreateFromFile);
  SYM(c, TfLiteInterpreterOptions* (*)(void), TfLiteInterpreterOptionsCreate);
  SYM(c, void (*)(TfLiteInterpreterOptions*, TfLiteDelegate*), TfLiteInterpreterOptionsAddDelegate);
  SYM(c, void (*)(TfLiteInterpreterOptions*, void (*)(void*, const char*, va_list), void*),
      TfLiteInterpreterOptionsSetErrorReporter);
  SYM(c, TfLiteInterpreter* (*)(const TfLiteModel*, const TfLiteInterpreterOptions*),
      TfLiteInterpreterCreate);
  SYM(c, TfLiteSignatureRunner* (*)(const TfLiteInterpreter*, const char*),
      TfLiteInterpreterGetSignatureRunner);
  SYM(c, TfLiteStatus (*)(TfLiteSignatureRunner*), TfLiteSignatureRunnerAllocateTensors);
  SYM(c, size_t (*)(const TfLiteSignatureRunner*), TfLiteSignatureRunnerGetInputCount);
  SYM(c, const char* (*)(const TfLiteSignatureRunner*, int32_t), TfLiteSignatureRunnerGetInputName);
  SYM(c, TfLiteTensor* (*)(TfLiteSignatureRunner*, const char*), TfLiteSignatureRunnerGetInputTensor);
  SYM(c, TfLiteStatus (*)(TfLiteSignatureRunner*), TfLiteSignatureRunnerInvoke);
  SYM(c, size_t (*)(const TfLiteSignatureRunner*), TfLiteSignatureRunnerGetOutputCount);
  SYM(c, const char* (*)(const TfLiteSignatureRunner*, int32_t), TfLiteSignatureRunnerGetOutputName);
  SYM(c, const TfLiteTensor* (*)(const TfLiteSignatureRunner*, const char*),
      TfLiteSignatureRunnerGetOutputTensor);
  SYM(c, void* (*)(const TfLiteTensor*), TfLiteTensorData);
  SYM(c, size_t (*)(const TfLiteTensor*), TfLiteTensorByteSize);
  SYM(c, TfLiteType (*)(const TfLiteTensor*), TfLiteTensorType);
  SYM(c, void (*)(TfLiteInterpreter*), TfLiteInterpreterDelete);

  printf("TfLiteVersion: %s\n", TfLiteVersion());

  TfLiteDelegate* delegate = NULL;
  void (*destroy)(TfLiteDelegate*) = NULL;
  if (use_flex) {
    void* flex = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!flex) { fprintf(stderr, "dlopen flex: %s\n", dlerror()); finish(1); }
    TfLiteDelegate* (*create)(const char* const*, const char* const*, size_t,
                              void (*)(const char*)) =
        (TfLiteDelegate* (*)(const char* const*, const char* const*, size_t,
                             void (*)(const char*)))dlsym(flex, "tflite_plugin_create_delegate");
    destroy = (void (*)(TfLiteDelegate*))dlsym(flex, "tflite_plugin_destroy_delegate");
    if (!create || !destroy) { fprintf(stderr, "flex plugin entry points missing\n"); finish(1); }
    delegate = create(NULL, NULL, 0, NULL);
    printf("flex delegate: %p\n", (void*)delegate);
    if (!delegate) finish(1);
  } else {
    printf("control run without the flex delegate\n");
  }

  TfLiteModel* model = TfLiteModelCreateFromFile(argv[3]);
  if (!model) { fprintf(stderr, "model load failed\n"); finish(1); }
  TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetErrorReporter(opts, reporter, NULL);
  if (delegate) TfLiteInterpreterOptionsAddDelegate(opts, delegate);
  TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, opts);
  printf("interpreter: %p\n", (void*)interp);
  if (!interp) finish(1);

  TfLiteSignatureRunner* r = TfLiteInterpreterGetSignatureRunner(interp, "train");
  if (!r) { fprintf(stderr, "no 'train' signature\n"); finish(1); }
  TfLiteStatus st = TfLiteSignatureRunnerAllocateTensors(r);
  printf("AllocateTensors: %d\n", st);
  if (st != 0) finish(1);

  size_t n_in = TfLiteSignatureRunnerGetInputCount(r);
  for (size_t i = 0; i < n_in; i++) {
    const char* name = TfLiteSignatureRunnerGetInputName(r, (int32_t)i);
    TfLiteTensor* t = TfLiteSignatureRunnerGetInputTensor(r, name);
    size_t bytes = TfLiteTensorByteSize(t);
    void* data = TfLiteTensorData(t);
    if (data && bytes) memset(data, 0, bytes);
    printf("input %s: type %d, %zu bytes, zero-filled\n", name, TfLiteTensorType(t), bytes);
  }

  st = TfLiteSignatureRunnerInvoke(r);
  printf("Invoke: %d\n", st);
  if (st != 0) finish(1);

  int bad = 0;
  size_t n_out = TfLiteSignatureRunnerGetOutputCount(r);
  for (size_t i = 0; i < n_out; i++) {
    const char* name = TfLiteSignatureRunnerGetOutputName(r, (int32_t)i);
    const TfLiteTensor* t = TfLiteSignatureRunnerGetOutputTensor(r, name);
    size_t bytes = TfLiteTensorByteSize(t);
    const void* data = TfLiteTensorData(t);
    printf("output %s: type %d, %zu bytes", name, TfLiteTensorType(t), bytes);
    if (TfLiteTensorType(t) == 1 && data && bytes >= sizeof(float)) {
      float v = ((const float*)data)[0];
      printf(", first value %g", v);
      if (!isfinite(v)) bad = 1;
    }
    printf("\n");
  }
  // Exercise the destroy entry point: the interpreter goes first, as it
  // references the delegate.
  TfLiteInterpreterDelete(interp);
  if (destroy && delegate) {
    destroy(delegate);
    printf("flex delegate destroyed\n");
  }
  printf(bad ? "RESULT: non-finite output\n" : "RESULT: train step ran\n");
  finish(bad);
  return 0;
}
