/// Native build of `package:flutter_litert`.
///
/// This barrel resolves the whole API against the native (`dart:ffi`)
/// implementation unconditionally. The portable `flutter_litert.dart` barrel
/// resolves to the WASM-safe web surface by default (so the package stays
/// WASM-compatible), which means symbols whose signatures use native-only types
/// (`Isolate`, `SendPort`, `File`, the native `Interpreter`, ...) cannot be
/// reached through it during static analysis.
///
/// Import this from code that only runs on native platforms (Android, iOS,
/// macOS, Windows, Linux) and needs that API, e.g. `IsolateWorkerBase`,
/// `IsolateRpcClient`, `InterpreterPool`, `ModelCheckpoint`, or
/// `TensorFloat32Views`. Do not import it together with `flutter_litert.dart`
/// in the same library: both define `Interpreter` and friends, which would be
/// an ambiguous import.
library;

export 'src/all_common.dart';
export 'src/all_native.dart';
