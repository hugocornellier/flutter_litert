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
  # These dylibs are small enough (~13 MB) to ship in the pub package,
  # so they are always present — both for app builds (CocoaPods) and `flutter test`.
  # Dylibs live inside the SPM source tree so both SPM and CocoaPods share one copy.
  resources = [
    'flutter_litert/Sources/flutter_litert/Resources/libtensorflowlite_c-mac.dylib',
    'flutter_litert/Sources/flutter_litert/Resources/libtflite_custom_ops.dylib',
  ]

  # Metal GPU delegate auto-bundling: if the developer has previously called
  # GpuDelegate.download(), detect the cached library and copy it into
  # Resources so it gets bundled with the app. No-op if not downloaded.
  metal_lib = 'libtensorflowlite_gpu-mac.dylib'
  metal_cache = File.expand_path("~/Library/Caches/flutter_litert/#{metal_lib}")
  metal_res = File.join(__dir__, 'flutter_litert', 'Sources', 'flutter_litert', 'Resources')
  metal_dest = File.join(metal_res, metal_lib)

  if File.exist?(metal_cache) && !File.exist?(metal_dest)
    puts "[flutter_litert] Bundling Metal GPU delegate from cache (#{metal_cache})..."
    FileUtils.cp(metal_cache, metal_dest)
  end

  if File.exist?(metal_dest)
    resources << "flutter_litert/Sources/flutter_litert/Resources/#{metal_lib}"
  end

  s.resources = resources
end
