#include "flutter_litert_gpu_shim.h"

// Share the Metal-accelerator registration shim used by the CocoaPods
// channel. Both channels ship LiteRT Next as framework-wrapped bundles
// (loose bare dylibs in Frameworks/ are rejected by App Store validation,
// issue #15), so both rely on the shim's exported
// LiteRtRegisterGpuAccelerator instead of the runtime's bare-dylib
// file-name scan.
#include "../../../Classes/litert_gpu_accelerator_shim.c"
