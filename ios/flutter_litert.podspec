#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_litert.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_litert'
  s.version          = '0.0.1'
  s.summary          = 'LiteRT (formerly TensorFlow Lite) plugin for Flutter apps.'
  s.description      = <<-DESC
LiteRT (formerly TensorFlow Lite) plugin for Flutter apps.
                       DESC
  s.homepage         = 'https://github.com/hugocornellier/flutter_litert'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Hugo Cornellier' => 'hugo@hugocornellier.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }

  # Include Swift plugin and forwarder C file (which #includes the actual sources)
  s.source_files = 'Classes/**/*'

  # Preserve paths for header includes (these won't be compiled, just available for #include)
  s.preserve_paths = '../src/tensorflow_lite/**/*.h', '../src/custom_ops/**/*.h'

  s.dependency 'Flutter'

  tflite_version = '2.17.0'
  s.dependency 'TensorFlowLiteSwift', tflite_version
  s.dependency 'TensorFlowLiteSwift/Metal', tflite_version
  s.dependency 'TensorFlowLiteSwift/CoreML', tflite_version

  s.platform = :ios, '12.0'
  s.static_framework = true

  # Flutter.framework does not contain a i386 slice.
  # Define TFLITE_USE_FRAMEWORK_HEADERS for iOS to use TensorFlowLiteC framework headers
  # GCC_SYMBOLS_PRIVATE_EXTERN=NO ensures C symbols are exported for FFI lookup
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) TFLITE_USE_FRAMEWORK_HEADERS=1',
    'GCC_SYMBOLS_PRIVATE_EXTERN' => 'NO',
    'HEADER_SEARCH_PATHS' => '"${PODS_TARGET_SRCROOT}/../src" "${PODS_TARGET_SRCROOT}/../src/custom_ops"'
  }

  # -ObjC forces the linker to load all ObjC classes and categories
  # This also pulls in C code that ObjC classes depend on
  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => '$(inherited) -ObjC'
  }
  s.swift_version = '5.0'
end
