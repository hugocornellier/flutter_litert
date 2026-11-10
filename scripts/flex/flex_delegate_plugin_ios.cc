// flutter_litert: Flex delegate plugin entry points for the iOS static
// framework (TensorFlowLiteFlex). Copied into
// tensorflow/lite/delegates/flex/ by .github/workflows/build-flex-ios.yml.
//
// iOS cannot dlopen third-party libraries, so the delegate is linked into the
// app and Dart finds these two symbols through DynamicLibrary.process().
#include "tensorflow/lite/delegates/flex/delegate.h"
#include <map>
#include <mutex>

namespace {
// FlexDelegate::Create() returns a TfLiteDelegateUniquePtr with a custom
// deleter. We must preserve the unique_ptr so the deleter runs on destroy.
std::mutex g_mu;
std::map<TfLiteDelegate*, tflite::TfLiteDelegateUniquePtr> g_delegates;
}

extern "C" {

__attribute__((visibility("default")))
TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*)) {
  auto delegate = tflite::FlexDelegate::Create();
  auto* raw = delegate.get();
  std::lock_guard<std::mutex> lock(g_mu);
  g_delegates.emplace(raw, std::move(delegate));
  return raw;
}

__attribute__((visibility("default")))
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_delegates.erase(delegate);
}

}  // extern "C"
