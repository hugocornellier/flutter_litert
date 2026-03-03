#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_litert.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_litert'
  s.version          = '0.0.1'
  s.summary          = 'LiteRT (formerly TensorFlow Lite) for Flutter with custom ops support.'
  s.description      = <<-DESC
LiteRT (formerly TensorFlow Lite) Flutter plugin with MediaPipe custom operations support.
                       DESC
  s.homepage         = 'https://github.com/hugocornellier/flutter_litert'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Hugo Cornellier' => 'hugo@hugocornellier.com' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'

  s.platform = :osx, '10.11'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'

  # Bundle the TFLite libraries as resources.
  # These dylibs are small enough to ship in the pub package,
  # so they are always present — both for app builds (CocoaPods) and `flutter test`.
  # Dylibs live inside the SPM source tree so both SPM and CocoaPods share one copy.
  s.resources = [
    'flutter_litert/Sources/flutter_litert/Resources/libtensorflowlite_c-mac.dylib',
    'flutter_litert/Sources/flutter_litert/Resources/libtflite_custom_ops.dylib',
    'flutter_litert/Sources/flutter_litert/Resources/libtensorflowlite_gpu-mac.dylib',
    'flutter_litert/Sources/flutter_litert/Resources/libtensorflowlite_coreml-mac.dylib',
  ]
end
