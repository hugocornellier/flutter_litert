/*
 * Copyright 2025 flutter_litert authors.
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
import 'dart:io';
import 'dart:typed_data';

import '../native/interpreter.dart';
import '../tensor_type.dart';
import 'byte_conversion_utils_shared.dart';

/// Dart-side weight persistence for on-device training models.
///
/// Saves and restores model weights to disk using the `get_weights` and
/// `set_weights` signatures that training models expose. This is exported on
/// native platforms only and does not require the Flex delegate or any native
/// library beyond the base TFLite runtime.
///
/// ## Binary checkpoint format (`.flwt`)
///
/// ```
/// [4B magic "FLWT"] [1B version] [4B tensor count N]
/// Per tensor:
///   [4B name length] [name UTF-8] [4B TfLiteType] [4B rank]
///   [rank*4B shape dims] [4B data byte count] [data bytes]
/// ```
///
/// ## Example
///
/// ```dart
/// final interpreter = Interpreter.fromFile(modelFile);
///
/// // Train...
/// final trainRunner = interpreter.getSignatureRunner('train');
/// for (int i = 0; i < 100; i++) {
///   trainRunner.run({'x': [[1.0]], 'y': [[2.0]]}, {'loss': Float32List(1)});
/// }
/// trainRunner.close();
///
/// // Save weights to disk
/// await ModelCheckpoint.save(interpreter, File('model.flwt'));
///
/// // Later, restore into a fresh interpreter
/// final fresh = Interpreter.fromFile(modelFile);
/// await ModelCheckpoint.restore(fresh, File('model.flwt'));
/// ```
class ModelCheckpoint {
  ModelCheckpoint._();

  static const List<int> _magic = [0x46, 0x4C, 0x57, 0x54]; // "FLWT"
  static const int _version = 0x01;

  /// Saves the current model weights to [file].
  ///
  /// The [interpreter] must have a `get_weights` signature. Weights are read
  /// from the model and serialized in the `.flwt` binary format.
  ///
  /// The write is atomic, a temporary file is written first, then renamed.
  static Future<void> save(Interpreter interpreter, File file) async {
    if (!interpreter.signatureKeys.contains('get_weights')) {
      throw ArgumentError('Model does not have a "get_weights" signature.');
    }

    final runner = interpreter.getSignatureRunner('get_weights');
    try {
      runner.allocateTensors();
      runner.invoke();

      final tensors = <_TensorRecord>[];
      for (final name in runner.outputNames) {
        final tensor = runner.getOutputTensor(name);
        tensors.add(
          _TensorRecord(
            name: name,
            type: tensor.type.value,
            shape: tensor.shape,
            data: Uint8List.fromList(tensor.data),
          ),
        );
      }

      final builder = BytesBuilder(copy: false);
      // Header
      builder.add(Uint8List.fromList(_magic));
      builder.addByte(_version);
      builder.add(_uint32LE(tensors.length));

      // Tensor records
      for (final t in tensors) {
        final nameBytes = utf8.encode(t.name);
        builder.add(_uint32LE(nameBytes.length));
        builder.add(Uint8List.fromList(nameBytes));
        builder.add(_int32LE(t.type));
        builder.add(_uint32LE(t.shape.length));
        for (final dim in t.shape) {
          builder.add(_int32LE(dim));
        }
        builder.add(_uint32LE(t.data.length));
        builder.add(t.data);
      }

      // Atomic write
      final tmpFile = File('${file.path}.tmp');
      await tmpFile.writeAsBytes(builder.takeBytes(), flush: true);
      await tmpFile.rename(file.path);
    } finally {
      runner.close();
    }
  }

  /// Restores model weights from [file] into [interpreter].
  ///
  /// The [interpreter] must have a `set_weights` signature. The file must be
  /// a valid `.flwt` checkpoint created by [save].
  static Future<void> restore(Interpreter interpreter, File file) async {
    if (!interpreter.signatureKeys.contains('set_weights')) {
      throw ArgumentError('Model does not have a "set_weights" signature.');
    }

    final bytes = await file.readAsBytes();
    final records = _parseCheckpoint(bytes);

    final runner = interpreter.getSignatureRunner('set_weights');
    try {
      final inputs = <String, Object>{};
      for (final record in records) {
        final tensorType = TensorType.fromValue(record.type);
        inputs[record.name] = ByteConversionUtils.convertBytesToObject(
          record.data,
          tensorType,
          record.shape,
        );
      }
      runner.run(inputs, {});
    } finally {
      runner.close();
    }
  }

  /// Returns a map of tensor name to shape for every tensor in a checkpoint
  /// file. Reads and parses the entire file, then discards the tensor data,
  /// retaining only names and shapes. Useful for debugging and validation.
  static Future<Map<String, List<int>>> inspect(File file) async {
    final bytes = await file.readAsBytes();
    final records = _parseCheckpoint(bytes);
    return {for (final r in records) r.name: r.shape};
  }

  // ---------------------------------------------------------------------------
  // Binary format helpers
  // ---------------------------------------------------------------------------

  static List<_TensorRecord> _parseCheckpoint(Uint8List bytes) {
    final bd = ByteData.view(bytes.buffer, bytes.offsetInBytes, bytes.length);
    int offset = 0;

    // Magic
    if (bytes.length < 9 ||
        bytes[0] != _magic[0] ||
        bytes[1] != _magic[1] ||
        bytes[2] != _magic[2] ||
        bytes[3] != _magic[3]) {
      throw FormatException('Invalid checkpoint file: bad magic bytes.');
    }
    offset += 4;

    // Version
    final version = bytes[offset];
    if (version != _version) {
      throw FormatException(
        'Unsupported checkpoint version: $version (expected $_version).',
      );
    }
    offset += 1;

    // Tensor count
    final tensorCount = bd.getUint32(offset, Endian.little);
    offset += 4;

    final records = <_TensorRecord>[];
    for (int i = 0; i < tensorCount; i++) {
      // Name
      final nameLen = bd.getUint32(offset, Endian.little);
      offset += 4;
      final name = utf8.decode(bytes.sublist(offset, offset + nameLen));
      offset += nameLen;

      // Type
      final type = bd.getInt32(offset, Endian.little);
      offset += 4;

      // Shape
      final rank = bd.getUint32(offset, Endian.little);
      offset += 4;
      final shape = <int>[];
      for (int d = 0; d < rank; d++) {
        shape.add(bd.getInt32(offset, Endian.little));
        offset += 4;
      }

      // Data
      final dataLen = bd.getUint32(offset, Endian.little);
      offset += 4;
      final data = Uint8List.fromList(bytes.sublist(offset, offset + dataLen));
      offset += dataLen;

      records.add(
        _TensorRecord(name: name, type: type, shape: shape, data: data),
      );
    }

    return records;
  }

  static Uint8List _uint32LE(int value) {
    final bd = ByteData(4)..setUint32(0, value, Endian.little);
    return bd.buffer.asUint8List();
  }

  static Uint8List _int32LE(int value) {
    final bd = ByteData(4)..setInt32(0, value, Endian.little);
    return bd.buffer.asUint8List();
  }
}

class _TensorRecord {
  final String name;
  final int type;
  final List<int> shape;
  final Uint8List data;

  _TensorRecord({
    required this.name,
    required this.type,
    required this.shape,
    required this.data,
  });
}
