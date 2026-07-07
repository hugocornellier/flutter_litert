import Flutter
import UIKit
import flutter_litert_custom_ops
import flutter_litert_delegate_symbols
import flutter_litert_gpu_shim

public class FlutterLitertPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    FlutterLitertRetainFfiSymbols()
    FlutterLitertRetainCustomOps()
    // Keep the CompiledModel Metal-accelerator registration shim linked; the
    // LiteRT runtime resolves it only via dlsym(RTLD_DEFAULT).
    FlutterLitertRetainLiteRtGpuShim()
    let channel = FlutterMethodChannel(name: "flutter_litert", binaryMessenger: registrar.messenger())
    let instance = FlutterLitertPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "getPlatformVersion":
      result("iOS " + UIDevice.current.systemVersion)
    default:
      result(FlutterMethodNotImplemented)
    }
  }
}
