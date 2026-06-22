import 'dart:typed_data';

/// Web: `dart:isolate` (and thus `TransferableTypedData`) is unavailable, and
/// detectors run on the main isolate, so there is no boundary to cross. Return
/// the [bytes] unchanged as the payload carrier.
Object transferableBytes(Uint8List bytes) => bytes;
