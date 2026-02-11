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

  # Bundle the TFLite libraries including custom ops as resources
  # This ensures they are copied to the app bundle's Resources directory
  s.resources = ['libtensorflowlite_c-mac.dylib', 'libtflite_custom_ops.dylib']
end
