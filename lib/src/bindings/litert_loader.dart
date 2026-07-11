/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:flutter_litert/src/delegates/delegate_library_loader.dart';

String? _litertRuntimeDir;

/// Directory containing the LiteRT runtime library that was opened, when known.
///
/// This is best effort for bare library-name loads because the platform dynamic
/// loader does not report the resolved path for `dlopen("libLiteRt...")`.
String? get litertRuntimeDir {
  litertDynamicLibrary;
  return _litertRuntimeDir;
}

/// Top-level finals are initialized lazily on first access in Dart.
final DynamicLibrary litertDynamicLibrary = () {
  if (Platform.isAndroid) {
    return _loadAndroidLibrary();
  }

  if (Platform.isIOS) {
    return _loadIosLibrary();
  }

  if (Platform.isMacOS || Platform.isLinux || Platform.isWindows) {
    return _loadDesktopLibrary();
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

DynamicLibrary _loadAndroidLibrary() {
  // LiteRT Next native libraries are bundled as jniLibs by
  // android/build.gradle.kts (downloaded at build time from the API-pinned
  // release), so the Android linker resolves bare sonames from the app's
  // native library directory.
  //
  // _litertRuntimeDir stays null on purpose: native libraries may be loaded
  // straight from the APK (extractNativeLibs=false), so there is no reliable
  // directory to scan. With no RuntimeLibraryDir option, the runtime's GPU
  // probe degenerates to dlopen'ing the bare accelerator sonames
  // (e.g. `libLiteRtClGlAccelerator.so`), which the Android linker resolves
  // from the same native library path, so an accelerator bundled for the
  // current ABI is still discoverable.
  _litertRuntimeDir = null;
  return DynamicLibrary.open('libLiteRt.so');
}

DynamicLibrary _loadIosLibrary() {
  // Both distribution channels embed the runtime as conventional framework
  // bundles (LiteRt.framework/LiteRt): CocoaPods rejects bare-dylib
  // xcframeworks at install time, and App Store validation rejects the
  // loose dylibs Xcode embeds for bare-dylib SwiftPM binary targets
  // (ITMS-90426, issue #15). The framework rename defeats LiteRT's
  // GPU-plugin file-name scan (`libLiteRtMetalAccelerator.dylib` inside the
  // directory passed as kLiteRtEnvOptionTagRuntimeLibraryDir,
  // litertRuntimeDir), so the Metal accelerator is registered by the
  // LiteRtRegisterGpuAccelerator shim in ios/Classes instead, found by the
  // runtime's RTLD_DEFAULT registration probe. The bare-dylib probe below
  // stays first for apps still built against pre-fix SwiftPM package
  // versions, where the loose-dylib layout (and the runtime's own file-name
  // scan) still applies.
  final frameworksDir =
      '${Directory(Platform.resolvedExecutable).parent.path}/Frameworks';
  final attemptedPaths = <String>[];
  final loaded = _probeLiteRtLibraryPaths(
    envVar: 'LITERT_LIB_PATH',
    paths: [
      '$frameworksDir/libLiteRt.dylib',
      '$frameworksDir/LiteRt.framework/LiteRt',
      'libLiteRt.dylib',
    ],
    attemptedPaths: attemptedPaths,
  );
  if (loaded != null) {
    // The accelerator scan expects the directory that holds the bare
    // accelerator dylib, which is the Frameworks directory in both layouts
    // that can provide it.
    _litertRuntimeDir = frameworksDir;
    return loaded.library;
  }

  // Fall back to the process image for apps that link LiteRT some other
  // way. The Metal accelerator cannot be auto-registered in this case
  // because the runtime directory is unknown.
  _litertRuntimeDir = null;
  return DynamicLibrary.process();
}

DynamicLibrary _loadDesktopLibrary() {
  final List<String> attemptedPaths = [];
  final libName = _desktopLibraryName;
  final packagePath = _resolvePackagePath('flutter_litert');

  final List<String> paths;
  if (Platform.isMacOS) {
    final appBundle = Directory(Platform.resolvedExecutable).parent.parent;
    final execDir = Directory(Platform.resolvedExecutable).parent.path;
    paths = [
      ...delegateBundlePaths(libName),
      '$execDir/$libName',
      '${appBundle.path}/Frameworks/$libName',
      if (packagePath != null)
        '$packagePath/macos/flutter_litert/Sources/flutter_litert/Resources/$libName',
      libName,
    ];
  } else {
    final execDir = Directory(Platform.resolvedExecutable).parent.path;
    final subDir = Platform.isLinux ? 'lib/' : '';
    final platformDir = Platform.isLinux ? 'linux/lib' : 'windows';
    paths = [
      '$execDir/$subDir$libName',
      if (packagePath != null) '$packagePath/$platformDir/$libName',
      libName,
    ];
  }

  final loaded = _probeLiteRtLibraryPaths(
    envVar: 'LITERT_LIB_PATH',
    paths: paths,
    attemptedPaths: attemptedPaths,
  );
  if (loaded != null) {
    _litertRuntimeDir = loaded.runtimeDir;
    return loaded.library;
  }

  throw UnsupportedError(
    'Failed to load LiteRT Next library. Attempted paths:\n'
    '${attemptedPaths.map((p) => '  - $p').join('\n')}\n\n'
    'Solutions:\n'
    '  1. For production apps: Ensure libLiteRt is bundled correctly\n'
    '  2. For testing: Set LITERT_LIB_PATH environment variable:\n'
    '     LITERT_LIB_PATH=/path/to/$libName flutter test\n'
    '  3. For testing: Ensure libraries exist in the project fallback locations',
  );
}

_LoadedLiteRtLibrary? _probeLiteRtLibraryPaths({
  required String envVar,
  required List<String> paths,
  required List<String> attemptedPaths,
}) {
  final envPath = Platform.environment[envVar];
  if (envPath != null && envPath.isNotEmpty) {
    attemptedPaths.add('$envVar: $envPath');
    final loaded = _tryOpenLiteRtLibrary(envPath, inferRuntimeDir: true);
    if (loaded != null) return loaded;
  }

  for (final path in paths) {
    attemptedPaths.add(path);
    final loaded = _tryOpenLiteRtLibrary(
      path,
      inferRuntimeDir: _hasDirectoryComponent(path),
    );
    if (loaded != null) return loaded;
  }

  return null;
}

_LoadedLiteRtLibrary? _tryOpenLiteRtLibrary(
  String path, {
  required bool inferRuntimeDir,
}) {
  try {
    final library = DynamicLibrary.open(path);
    final runtimeDir = inferRuntimeDir ? _runtimeDirForLibraryPath(path) : null;
    return _LoadedLiteRtLibrary(library, runtimeDir);
  } catch (_) {
    return null;
  }
}

String _runtimeDirForLibraryPath(String path) {
  final file = File(path);
  try {
    return File(file.resolveSymbolicLinksSync()).parent.path;
  } catch (_) {
    return file.absolute.parent.path;
  }
}

bool _hasDirectoryComponent(String path) {
  return path.contains('/') || path.contains(r'\');
}

final class _LoadedLiteRtLibrary {
  const _LoadedLiteRtLibrary(this.library, this.runtimeDir);

  final DynamicLibrary library;
  final String? runtimeDir;
}

String get _desktopLibraryName {
  if (Platform.isMacOS) return 'libLiteRt.dylib';
  if (Platform.isLinux) return 'libLiteRt.so';
  if (Platform.isWindows) return 'libLiteRt.dll';
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}

String? _resolvePackagePath(String packageName) {
  try {
    final configFile = File(
      '${Directory.current.path}/.dart_tool/package_config.json',
    );
    if (!configFile.existsSync()) return null;

    final config = jsonDecode(configFile.readAsStringSync()) as Map;
    final packages = config['packages'] as List?;
    if (packages == null) return null;

    for (final pkg in packages) {
      if (pkg is Map && pkg['name'] == packageName) {
        final rootUri = pkg['rootUri'] as String?;
        if (rootUri == null) continue;

        if (rootUri.startsWith('file://')) {
          return Uri.parse(rootUri).toFilePath();
        }
        final resolved = configFile.parent.uri.resolve(rootUri);
        return resolved.toFilePath();
      }
    }
  } catch (_) {
    // Best-effort fallback for local test runs.
  }
  return null;
}
