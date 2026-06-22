import 'dart:isolate';
import 'dart:typed_data';

/// Native: wrap [bytes] in a [TransferableTypedData] so they cross an isolate
/// boundary with zero copy. Materialize on the isolate side with
/// `(message['bytes'] as TransferableTypedData).materialize().asUint8List()`.
Object transferableBytes(Uint8List bytes) =>
    TransferableTypedData.fromList([bytes]);
