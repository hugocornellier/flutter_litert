// Objective-C wrapper for loading custom ops

#import "TfliteCustomOpsLoader.h"

// Forward declare the C function
extern void* TfLiteFlutter_RegisterConvolution2DTransposeBias(void);

// Static variable to hold the registration - this prevents the linker from
// stripping the symbol since it's stored in a global variable
static void* _customOpsRegistration = NULL;

@implementation TfliteCustomOpsLoader

+ (void)loadCustomOps {
    // Call and store the result to prevent dead code elimination
    _customOpsRegistration = TfLiteFlutter_RegisterConvolution2DTransposeBias();

    // Log to ensure this runs and prevent optimization
    NSLog(@"TFLite custom ops loaded: %p", _customOpsRegistration);
}

@end
