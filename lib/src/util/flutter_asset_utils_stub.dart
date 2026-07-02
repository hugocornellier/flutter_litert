import 'dart:typed_data';

/// Non-Flutter fallback for [loadAssetBytes].
///
/// Selected by the conditional import in `native/interpreter.dart` when
/// `dart:ui` is unavailable (plain Dart VM tools such as benchmarks). Flutter
/// apps always resolve to the real implementation in
/// `flutter_asset_utils.dart`.
Future<Uint8List> loadAssetBytes(String assetFileName) =>
    throw UnsupportedError(
      'Interpreter.fromAsset requires the Flutter runtime; '
      'use Interpreter.fromFile or Interpreter.fromBuffer instead.',
    );
