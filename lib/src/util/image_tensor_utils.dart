import 'dart:typed_data';

/// 256-entry byte-to-float tables for the two normalizations. Storing the
/// float32-rounded value keeps results bit-identical to computing
/// `byte * scale + offset` per element while replacing the int-to-double
/// conversion, multiply, and add with one indexed load.
final Float32List _lutUnit = _buildLut(1.0 / 255.0, 0.0);
final Float32List _lutSigned = _buildLut(1.0 / 127.5, -1.0);

Float32List _buildLut(double scale, double offset) {
  final lut = Float32List(256);
  for (var i = 0; i < 256; i++) {
    lut[i] = i * scale + offset;
  }
  return lut;
}

/// Converts BGR bytes to a flat Float32List with `0.0..1.0` normalization.
///
/// Performs a BGR-to-RGB channel swap and divides each value by 255.
///
/// Parameters:
/// - [bytes]: Raw BGR image bytes (length must be totalPixels * 3)
/// - [totalPixels]: Total number of pixels (width * height)
/// - [buffer]: Optional pre-allocated Float32List of length totalPixels * 3 to reuse
///
/// Returns a flat Float32List with normalized RGB pixel values in `0.0..1.0`.
Float32List bgrBytesToRgbFloat32({
  required Uint8List bytes,
  required int totalPixels,
  Float32List? buffer,
}) => _bgrToRgbFloat32(
  bytes: bytes,
  totalPixels: totalPixels,
  lut: _lutUnit,
  buffer: buffer,
);

/// Converts BGR bytes to a flat Float32List with `-1.0..1.0` normalization.
///
/// Performs a BGR-to-RGB channel swap and normalizes via (value / 127.5) - 1.0,
/// as used by models expecting signed normalized input.
///
/// Parameters:
/// - [bytes]: Raw BGR image bytes (length must be totalPixels * 3)
/// - [totalPixels]: Total number of pixels (width * height)
/// - [buffer]: Optional pre-allocated Float32List of length totalPixels * 3 to reuse
///
/// Returns a flat Float32List with normalized RGB pixel values in `-1.0..1.0`.
Float32List bgrBytesToSignedFloat32({
  required Uint8List bytes,
  required int totalPixels,
  Float32List? buffer,
}) => _bgrToRgbFloat32(
  bytes: bytes,
  totalPixels: totalPixels,
  lut: _lutSigned,
  buffer: buffer,
);

Float32List _bgrToRgbFloat32({
  required Uint8List bytes,
  required int totalPixels,
  required Float32List lut,
  Float32List? buffer,
}) {
  final int size = totalPixels * 3;
  final Float32List tensor = buffer ?? Float32List(size);

  for (int i = 0, j = 0; i < size && j < size; i += 3, j += 3) {
    tensor[j] = lut[bytes[i + 2]];
    tensor[j + 1] = lut[bytes[i + 1]];
    tensor[j + 2] = lut[bytes[i]];
  }
  return tensor;
}

/// Converts RGBA pixel data to a normalized RGB Float32List tensor in-place.
///
/// Normalizes each channel to `0.0..1.0`. The alpha channel is discarded. This
/// is the RGBA-input sibling of [bgrBytesToRgbFloat32], for pixel data that
/// already arrives in RGBA order (e.g. a web canvas readback).
///
/// Parameters:
/// - [rgbaData]: RGBA pixel bytes (length must be a multiple of 4)
/// - [output]: Pre-allocated buffer (length must be rgbaData.length * 3 ~/ 4)
void rgbaToRgbFloat32(Uint8List rgbaData, Float32List output) {
  final Float32List lut = _lutUnit;
  int dst = 0;
  for (int src = 0; src < rgbaData.length; src += 4) {
    output[dst++] = lut[rgbaData[src]];
    output[dst++] = lut[rgbaData[src + 1]];
    output[dst++] = lut[rgbaData[src + 2]];
  }
}

/// Converts RGBA pixel data to a signed RGB Float32List tensor in-place.
///
/// Normalizes each channel via `(value / 127.5) - 1.0` into `-1.0..1.0`, as
/// expected by models trained on signed normalized input (e.g. MediaPipe). The
/// alpha channel is discarded. This is the RGBA-input sibling of
/// [bgrBytesToSignedFloat32].
///
/// Parameters:
/// - [rgbaData]: RGBA pixel bytes (length must be a multiple of 4)
/// - [output]: Pre-allocated buffer (length must be rgbaData.length * 3 ~/ 4)
void rgbaToSignedRgbFloat32(Uint8List rgbaData, Float32List output) {
  final Float32List lut = _lutSigned;
  int dst = 0;
  for (int src = 0; src < rgbaData.length; src += 4) {
    output[dst++] = lut[rgbaData[src]];
    output[dst++] = lut[rgbaData[src + 1]];
    output[dst++] = lut[rgbaData[src + 2]];
  }
}

/// Fills a 4D NHWC tensor in-place from raw BGR bytes with a BGR-to-RGB channel swap.
///
/// The [tensor] must already be allocated as `[1][height][width][3]`.
/// Use [scale] and [offset] to control normalization:
/// - For `0.0..1.0`: scale=1/255, offset=0.0 (default)
/// - For `-1.0..1.0`: scale=1/127.5, offset=-1.0
///
/// Parameters:
/// - [bytes]: Raw BGR image bytes (length must be width * height * 3)
/// - [tensor]: Pre-allocated 4D tensor `[1][height][width][3]` to fill
/// - [width]: Image width in pixels
/// - [height]: Image height in pixels
/// - [scale]: Multiplier applied to each byte value before adding offset
/// - [offset]: Value added after scaling
void fillNHWC4DFromBgrBytes({
  required Uint8List bytes,
  required List<List<List<List<double>>>> tensor,
  required int width,
  required int height,
  double scale = 1.0 / 255.0,
  double offset = 0.0,
}) {
  int byteIndex = 0;

  for (int y = 0; y < height; y++) {
    final List<List<double>> row = tensor[0][y];
    for (int x = 0; x < width; x++) {
      final List<double> pixel = row[x];
      pixel[0] = bytes[byteIndex + 2] * scale + offset;
      pixel[1] = bytes[byteIndex + 1] * scale + offset;
      pixel[2] = bytes[byteIndex] * scale + offset;
      byteIndex += 3;
    }
  }
}
