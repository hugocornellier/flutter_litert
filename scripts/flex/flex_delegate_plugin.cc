// flutter_litert: C wrapper for the TFLite Flex delegate external plugin
// interface, used for the desktop shared library (macOS dylib). Copied into
// tensorflow/lite/delegates/flex/ by .github/workflows/build-flex-macos.yml.
//
// Exports the two symbols flutter_litert's FlexDelegate loader looks up:
//   tflite_plugin_create_delegate
//   tflite_plugin_destroy_delegate
//
// FlexDelegate::Create() returns TfLiteDelegateUniquePtr, whose deleter is
// TfLiteDelegateFactory::DeleteSimpleDelegate. The destroy entry point must
// use that deleter, never plain `delete`.

#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

extern "C" {

TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*)) {
  // Ownership moves to the caller; freed by tflite_plugin_destroy_delegate.
  auto delegate = tflite::FlexDelegate::Create();
  return delegate.release();
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

}  // extern "C"
