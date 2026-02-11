// Objective-C wrapper for loading custom ops
// This ensures the C symbols are linked into the binary

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TfliteCustomOpsLoader : NSObject

/// Call this method to ensure custom ops are linked into the binary.
/// The actual FFI lookup still happens via Dart, but this forces
/// the linker to include the symbols.
+ (void)loadCustomOps;

@end

NS_ASSUME_NONNULL_END
