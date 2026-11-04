package com.hugocornellier.flutter_litert

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.annotation.NonNull

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

/** FlutterLitertPlugin */
class FlutterLitertPlugin: FlutterPlugin, MethodCallHandler {
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private lateinit var channel : MethodChannel
  private lateinit var context: Context
  private val mainHandler = Handler(Looper.getMainLooper())

  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    context = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "flutter_litert")
    channel.setMethodCallHandler(this)
  }

  override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
    when (call.method) {
      "getPlatformVersion" -> result.success("Android ${android.os.Build.VERSION.RELEASE}")
      "initializeRuntime" -> {
        val enableGpu = call.argument<Boolean>("enableGpu") ?: false
        val enableAutomaticDownload = call.argument<Boolean>("enableAutomaticDownload") ?: true
        val timeoutMillis = call.argument<Number>("timeoutMillis")?.toLong() ?: 300_000L
        // Completes on a background thread; MethodChannel replies must be sent
        // from the platform thread.
        val callback = object : PlayServicesRuntime.Callback {
          override fun onComplete(status: Map<String, Any?>) {
            mainHandler.post { result.success(status) }
          }

          override fun onError(error: String) {
            mainHandler.post { result.error("initializeRuntime", error, null) }
          }
        }
        // Loading the class loads libflutter_litert_runtime.so, which throws an
        // Error rather than an Exception when it fails. Reply either way, so the
        // Dart call never waits forever.
        try {
          PlayServicesRuntime.initialize(
            context, enableGpu, enableAutomaticDownload, timeoutMillis, callback)
        } catch (t: Throwable) {
          result.error("initializeRuntime", t.toString(), null)
        }
      }
      else -> result.notImplemented()
    }
  }

  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
  }
}
