import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_litert_platform_interface.dart';

/// An implementation of [FlutterLitertPlatform] that uses method channels.
class MethodChannelFlutterLitert extends FlutterLitertPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_litert');

  @override
  Future<String?> getPlatformVersion() async {
    final version =
        await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
