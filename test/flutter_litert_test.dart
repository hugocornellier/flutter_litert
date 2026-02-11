import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert/flutter_litert_platform_interface.dart';
import 'package:flutter_litert/flutter_litert_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterLitertPlatform
    with MockPlatformInterfaceMixin
    implements FlutterLitertPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final FlutterLitertPlatform initialPlatform = FlutterLitertPlatform.instance;

  test('$MethodChannelFlutterLitert is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFlutterLitert>());
  });

  test('getPlatformVersion', () async {
    expect(version, '42');
  });
}
