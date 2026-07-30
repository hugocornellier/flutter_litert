import 'dart:ffi';
import 'dart:io';

/// Returns the standard app-bundle search paths for a macOS delegate dylib
/// bundled inside `flutter_litert`'s resource directories.
List<String> delegateBundlePaths(String libName) {
  final appBundle = Directory(Platform.resolvedExecutable).parent.parent;
  return [
    '${appBundle.path}/Resources/$libName',
    '${appBundle.path}/Frameworks/flutter_litert.framework/Versions/A/Resources/$libName',
    '${appBundle.path}/Frameworks/flutter_litert.framework/Resources/$libName',
    '${appBundle.path}/Resources/flutter_litert_flutter_litert.bundle/Contents/Resources/$libName',
  ];
}

/// Tries to open a [DynamicLibrary] from [paths], optionally checking [envVar]
/// first. Appends all attempted entries to [attemptedPaths] for error
/// reporting. Returns the loaded library, or `null` if all attempts fail.
DynamicLibrary? probeLibraryPaths({
  String? envVar,
  required List<String> paths,
  required List<String> attemptedPaths,
}) => probeLibraryPathsWithPath(
  envVar: envVar,
  paths: paths,
  attemptedPaths: attemptedPaths,
)?.library;

/// Like [probeLibraryPaths], but also returns the exact candidate passed to
/// [DynamicLibrary.open].
///
/// This is useful for integration tests that must prove a production app loaded
/// its bundled resource rather than a package-checkout fallback.
({DynamicLibrary library, String path})? probeLibraryPathsWithPath({
  String? envVar,
  required List<String> paths,
  required List<String> attemptedPaths,
}) {
  if (envVar != null) {
    final envPath = Platform.environment[envVar];
    if (envPath != null && envPath.isNotEmpty) {
      attemptedPaths.add('$envVar: $envPath');
      try {
        return (library: DynamicLibrary.open(envPath), path: envPath);
      } catch (_) {}
    }
  }
  for (final path in paths) {
    attemptedPaths.add(path);
    try {
      return (library: DynamicLibrary.open(path), path: path);
    } catch (_) {}
  }
  return null;
}

/// Opens a delegate dylib on macOS using a standard search strategy:
/// 1. Check [getCached] for a previously loaded library
/// 2. Try environment variable override via [envVar]
/// 3. Iterate [bundlePaths] in order
/// 4. Throw UnsupportedError with attempted paths if none found
DynamicLibrary openDelegateLibrary({
  required String envVar,
  required List<String> bundlePaths,
  required String description,
  required DynamicLibrary? Function() getCached,
  required void Function(DynamicLibrary) setCached,
}) {
  final cached = getCached();
  if (cached != null) return cached;

  final List<String> attemptedPaths = [];
  final lib = probeLibraryPaths(
    envVar: envVar,
    paths: bundlePaths,
    attemptedPaths: attemptedPaths,
  );
  if (lib != null) {
    setCached(lib);
    return lib;
  }

  throw UnsupportedError(
    '$description library not found. Attempted paths:\n'
    '${attemptedPaths.map((p) => '  - $p').join('\n')}\n\n'
    'Solutions:\n'
    '  1. Set $envVar environment variable to the library path\n'
    '  2. The $description requires macOS on Apple Silicon (arm64)\n',
  );
}
