/* Load a TensorFlow Lite C shared library, print TfLiteVersion() and run a
 * model through it, once on the default CPU path and once with XNNPack.
 *
 * Used by the build-tflite-c-*.yml workflows to verify a freshly built
 * library (and the committed one it replaces) without any TFLite headers.
 *
 * Usage: tflite_c_check <library> <expected version prefix> [model.tflite]
 * Exit code 0 only when the version matches and every run succeeds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static void* load_lib(const char* p) { return (void*)LoadLibraryA(p); }
static void* sym(void* h, const char* n) {
  return (void*)GetProcAddress((HMODULE)h, n);
}
#else
#include <dlfcn.h>
static void* load_lib(const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void* sym(void* h, const char* n) { return dlsym(h, n); }
#endif

typedef const char* (*version_fn)(void);
typedef void* (*model_create_fn)(const char*);
typedef void (*model_delete_fn)(void*);
typedef void* (*opts_create_fn)(void);
typedef void (*opts_delete_fn)(void*);
typedef void (*opts_add_delegate_fn)(void*, void*);
typedef void* (*interp_create_fn)(const void*, const void*);
typedef void (*interp_delete_fn)(void*);
typedef int (*interp_status_fn)(void*);
typedef int (*interp_count_fn)(const void*);
typedef void* (*interp_input_fn)(const void*, int);
typedef size_t (*tensor_size_fn)(const void*);
typedef int (*tensor_copy_from_fn)(void*, const void*, size_t);
typedef void* (*xnn_create_fn)(const void*);
typedef void (*xnn_delete_fn)(void*);

static void* need(void* h, const char* n) {
  void* p = sym(h, n);
  if (!p) {
    fprintf(stderr, "missing export: %s\n", n);
    exit(3);
  }
  return p;
}

static int run_model(void* h, const char* model_path, int use_xnnpack) {
  model_create_fn model_create = (model_create_fn)need(h, "TfLiteModelCreateFromFile");
  model_delete_fn model_delete = (model_delete_fn)need(h, "TfLiteModelDelete");
  opts_create_fn opts_create = (opts_create_fn)need(h, "TfLiteInterpreterOptionsCreate");
  opts_delete_fn opts_delete = (opts_delete_fn)need(h, "TfLiteInterpreterOptionsDelete");
  opts_add_delegate_fn add_delegate =
      (opts_add_delegate_fn)need(h, "TfLiteInterpreterOptionsAddDelegate");
  interp_create_fn interp_create = (interp_create_fn)need(h, "TfLiteInterpreterCreate");
  interp_delete_fn interp_delete = (interp_delete_fn)need(h, "TfLiteInterpreterDelete");
  interp_status_fn allocate = (interp_status_fn)need(h, "TfLiteInterpreterAllocateTensors");
  interp_status_fn invoke = (interp_status_fn)need(h, "TfLiteInterpreterInvoke");
  interp_count_fn input_count =
      (interp_count_fn)need(h, "TfLiteInterpreterGetInputTensorCount");
  interp_input_fn input_tensor =
      (interp_input_fn)need(h, "TfLiteInterpreterGetInputTensor");
  tensor_size_fn byte_size = (tensor_size_fn)need(h, "TfLiteTensorByteSize");
  tensor_copy_from_fn copy_from =
      (tensor_copy_from_fn)need(h, "TfLiteTensorCopyFromBuffer");

  const char* label = use_xnnpack ? "xnnpack" : "default";
  void* model = model_create(model_path);
  if (!model) {
    fprintf(stderr, "[%s] TfLiteModelCreateFromFile failed\n", label);
    return 1;
  }
  void* opts = opts_create();
  void* xnn = NULL;
  xnn_delete_fn xnn_delete = NULL;
  if (use_xnnpack) {
    xnn_create_fn xnn_create = (xnn_create_fn)need(h, "TfLiteXNNPackDelegateCreate");
    xnn_delete = (xnn_delete_fn)need(h, "TfLiteXNNPackDelegateDelete");
    xnn = xnn_create(NULL);
    if (!xnn) {
      fprintf(stderr, "[%s] TfLiteXNNPackDelegateCreate failed\n", label);
      return 1;
    }
    add_delegate(opts, xnn);
  }
  void* interp = interp_create(model, opts);
  if (!interp) {
    fprintf(stderr, "[%s] TfLiteInterpreterCreate failed\n", label);
    return 1;
  }
  if (allocate(interp) != 0) {
    fprintf(stderr, "[%s] AllocateTensors failed\n", label);
    return 1;
  }
  int n = input_count(interp);
  for (int i = 0; i < n; ++i) {
    void* t = input_tensor(interp, i);
    size_t sz = byte_size(t);
    void* buf = calloc(sz ? sz : 1, 1);
    if (copy_from(t, buf, sz) != 0) {
      fprintf(stderr, "[%s] CopyFromBuffer failed on input %d\n", label, i);
      return 1;
    }
    free(buf);
  }
  int status = invoke(interp);
  printf("[%s] inputs=%d invoke status=%d\n", label, n, status);
  interp_delete(interp);
  if (xnn) xnn_delete(xnn);
  opts_delete(opts);
  model_delete(model);
  return status == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <library> <expected version prefix> [model]\n",
            argv[0]);
    return 2;
  }
  void* h = load_lib(argv[1]);
  if (!h) {
    fprintf(stderr, "cannot load %s\n", argv[1]);
    return 2;
  }
  version_fn version = (version_fn)need(h, "TfLiteVersion");
  const char* v = version();
  printf("TfLiteVersion() = %s\n", v);
  version_fn schema = (version_fn)sym(h, "TfLiteExtensionApisVersion");
  if (schema) printf("TfLiteExtensionApisVersion() = %s\n", schema());
  if (strncmp(v, argv[2], strlen(argv[2])) != 0) {
    fprintf(stderr, "version mismatch: got %s, expected %s*\n", v, argv[2]);
    return 1;
  }
  if (argc > 3) {
    if (run_model(h, argv[3], 0) != 0) return 1;
    if (run_model(h, argv[3], 1) != 0) return 1;
  }
  printf("OK\n");
  return 0;
}
