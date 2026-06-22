import 'dart:typed_data';

import '../isolate_rpc_error.dart' show IsolateRpcExactError;

/// Wire-prefix marking an isolate error as an undecodable-image failure, so the
/// main side can map it to a [FormatException]. Paired with [throwDecodeFailure]
/// (isolate side) and [rethrowOrFormatException] (main side).
const String decodeFailurePrefix = 'litert_decode_failure:';

/// Isolate-side: signal that image bytes could not be decoded.
///
/// Throws an [IsolateRpcExactError] whose verbatim wire message starts with
/// [decodeFailurePrefix]; the main side's [rethrowOrFormatException] maps it to
/// a [FormatException]. Use after an `imdecode` that returns an empty Mat.
Never throwDecodeFailure([
  String detail = 'Image bytes could not be decoded.',
]) {
  throw IsolateRpcExactError('$decodeFailurePrefix $detail');
}

/// Main-side: if [error] is a decode-failure signal (a [StateError] whose
/// message starts with [decodeFailurePrefix], as produced by [throwDecodeFailure]
/// in the isolate and re-wrapped by `IsolateRpcClient`), throw a
/// [FormatException] carrying the detail and optional [source] bytes; otherwise
/// rethrow [error] unchanged.
///
/// ```dart
/// try {
///   result = await worker.sendRequest('detect', params);
/// } catch (e) {
///   rethrowOrFormatException(e, imageBytes);
/// }
/// ```
Never rethrowOrFormatException(Object error, [Uint8List? source]) {
  final String msg = error is StateError ? error.message : error.toString();
  if (msg.startsWith(decodeFailurePrefix)) {
    final String detail = msg.substring(decodeFailurePrefix.length).trim();
    throw FormatException(
      detail.isEmpty ? 'Image bytes could not be decoded.' : detail,
      source,
    );
  }
  throw error;
}
