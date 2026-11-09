/*
 * Load a freshly built flex delegate library, create and destroy one
 * delegate, and print any version functions the library exports.
 *
 *   smoke_test <path to libtensorflowlite_flex-*.so|dll>
 *
 * Exit code 0 only when both plugin symbols resolve and create returns a
 * non-null delegate.
 */
#include <stddef.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_t;
static lib_t open_lib(const char* p) { return LoadLibraryA(p); }
static void* sym(lib_t l, const char* n) { return (void*)GetProcAddress(l, n); }
static void print_load_error(void) {
  fprintf(stderr, "LoadLibrary failed, error %lu\n", GetLastError());
}
#else
#include <dlfcn.h>
#include <unistd.h>
typedef void* lib_t;
static lib_t open_lib(const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void* sym(lib_t l, const char* n) { return dlsym(l, n); }
static void print_load_error(void) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); }
#endif

typedef void* (*create_fn)(const char* const*, const char* const*, size_t,
                           void (*)(const char*));
typedef void (*destroy_fn)(void*);
typedef const char* (*version_fn)(void);

static void report(const char* msg) { fprintf(stderr, "report_error: %s\n", msg); }

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <library>\n", argv[0]);
    return 2;
  }
  lib_t lib = open_lib(argv[1]);
  if (!lib) {
    print_load_error();
    return 1;
  }
  printf("loaded: %s\n", argv[1]);

  const char* version_syms[] = {"TfLiteVersion", "TF_Version", NULL};
  for (int i = 0; version_syms[i]; ++i) {
    version_fn f = (version_fn)sym(lib, version_syms[i]);
    if (f) {
      printf("%s() = %s\n", version_syms[i], f());
    } else {
      printf("%s: not exported\n", version_syms[i]);
    }
  }

  create_fn create = (create_fn)sym(lib, "tflite_plugin_create_delegate");
  destroy_fn destroy = (destroy_fn)sym(lib, "tflite_plugin_destroy_delegate");
  printf("tflite_plugin_create_delegate: %s\n", create ? "found" : "MISSING");
  printf("tflite_plugin_destroy_delegate: %s\n", destroy ? "found" : "MISSING");
  if (!create || !destroy) return 1;

  void* delegate = create(NULL, NULL, 0, report);
  printf("create -> %p\n", delegate);
  if (!delegate) return 1;
  destroy(delegate);
  printf("destroy -> ok\n");
  printf("SMOKE OK\n");
  fflush(stdout);
  /* Skip static destructors: TF's exit-time teardown is known to crash in
   * some hosts (see flex_save_restore_test), and that is not what this checks. */
#ifdef _WIN32
  TerminateProcess(GetCurrentProcess(), 0);
#else
  _exit(0);
#endif
}
