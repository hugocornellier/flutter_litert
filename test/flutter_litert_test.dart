import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

void main() {
  test('getPlatformVersion', () async {
    // Prefix, not equality: the TFLite version is what matters here, not the
    // build provenance. The TF v2.21.0 builds report plain "2.21.0"; the
    // earlier bazel-built macOS arm64 slice (see
    // doc/macos_transpose_conv_gap.md) reported a build suffix,
    // "2.20.0-dev0+selfbuilt".
    expect(version, startsWith('2.21.0'));
  });
}
