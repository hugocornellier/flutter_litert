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

  # FlexDelegate auto-bundling: if the developer has previously called
  # FlexDelegate.download(), detect the cached library and copy it into
  # Resources so it gets bundled with the app. No-op if not downloaded.
  # Skip if flutter_litert_flex is present (it bundles its own copy).
  flex_lib = 'libtensorflowlite_flex-mac.dylib'
  flex_cache = File.expand_path("~/Library/Caches/flutter_litert/#{flex_lib}")
  flex_res = File.join(__dir__, 'flutter_litert', 'Sources', 'flutter_litert', 'Resources')
  flex_dest = File.join(flex_res, flex_lib)

  # Detect if flutter_litert_flex plugin is installed (it handles bundling itself)
  flex_plugin_present = Dir.glob(File.join(__dir__, '..', 'flutter_litert_flex*')).any? ||
    Dir.glob(File.join(__dir__, '..', '.symlinks', 'plugins', 'flutter_litert_flex*')).any?

  if flex_plugin_present
    puts '[flutter_litert] flutter_litert_flex detected — skipping auto-bundle from cache.'
    # Clean up any previously cached copy to avoid shipping duplicates
    File.delete(flex_dest) if File.exist?(flex_dest)
  else
    if File.exist?(flex_cache) && !File.exist?(flex_dest)
      puts "[flutter_litert] Bundling FlexDelegate from cache (#{flex_cache})..."
      FileUtils.cp(flex_cache, flex_dest)
    end

    if File.exist?(flex_dest)
      resources << "flutter_litert/Sources/flutter_litert/Resources/#{flex_lib}"
    end
  end

  s.resources = resources
end
