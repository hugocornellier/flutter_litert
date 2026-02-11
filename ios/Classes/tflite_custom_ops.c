// Forwarder file that includes the custom ops implementation.
// This is necessary because CocoaPods doesn't support relative paths
// outside the pod directory in source_files.

// Include the actual implementation
#include "../../src/custom_ops/transpose_conv_bias.c"

// Force linker to include the custom ops symbol.
// This function is called from Swift to ensure the C code isn't stripped.
__attribute__((used))
void TfLiteFlutter_ForceLoadCustomOps(void) {
    // Reference the symbol to prevent linker from stripping it
    (void)TfLiteFlutter_RegisterConvolution2DTransposeBias;
}
