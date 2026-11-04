import '../android_runtime.dart';

/// The LiteRT runtime this app uses. Always [LiteRtRuntime.bundled] here.
LiteRtRuntime get androidRuntime => LiteRtRuntime.bundled;

/// Initializes the Android LiteRT runtime. A no-op off Android.
Future<LiteRtRuntimeStatus> initializeAndroidRuntime({
  bool enableGpu = false,
  bool enableAutomaticDownload = true,
  Duration timeout = const Duration(minutes: 5),
}) async {
  return const LiteRtRuntimeStatus.bundled();
}
