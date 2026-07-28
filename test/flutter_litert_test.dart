import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

void main() {
  test('getPlatformVersion', () async {
    // Prefix, not equality: the macOS arm64 slice is built from TF 2.20.0 with
    // bazel (to restore ruy multithreading, see
    // doc/macos_transpose_conv_gap.md) and so reports a build suffix,
    // "2.20.0-dev0+selfbuilt". The TFLite version is what matters here, not
    // the build provenance.
    expect(version, startsWith('2.20.0'));
  });
}
