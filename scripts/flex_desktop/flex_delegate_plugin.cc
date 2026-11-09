// C wrapper for the TFLite Flex delegate external plugin interface.
//
// Copied into tensorflow/lite/delegates/flex/ by prepare_tf.sh and built into
// libtensorflowlite_flex-linux.so / libtensorflowlite_flex-win.dll with
// --config=monolithic, so the library is self-contained.
//
// Exports the two symbols flutter_litert's FlexDelegate loader looks up:
//   tflite_plugin_create_delegate
//   tflite_plugin_destroy_delegate
//
// Symbol visibility comes from external_delegate_interface.h:
//   - Linux: __attribute__((visibility("default")))
//   - Windows: __declspec(dllexport), because the BUILD target defines
//     TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY (local_defines).
//
// FlexDelegate::Create() returns a TfLiteDelegateUniquePtr whose deleter is
// TfLiteDelegateFactory::DeleteSimpleDelegate. Destroy must use that deleter,
// never a plain delete of the TfLiteDelegate pointer.

#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

extern "C" {

TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys, const char* const* options_values,
    size_t num_options, void (*report_error)(const char*)) {
  // Ownership moves to the caller, who frees it with
  // tflite_plugin_destroy_delegate.
  auto delegate = tflite::FlexDelegate::Create();
  return delegate.release();
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

}  // extern "C"
