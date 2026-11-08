package com.hugocornellier.flutter_litert;

import android.content.Context;
import androidx.annotation.NonNull;
import java.util.HashMap;
import java.util.Map;

/**
 * Bundled-runtime counterpart of the Play services PlayServicesRuntime (flutterLitert.runtime
 * defaults to bundled). The bundled LiteRT runtime is packaged in the APK and needs no
 * initialization, so this reports the Interpreter and CompiledModel APIs as ready.
 */
final class PlayServicesRuntime {
  static final String MODE = "bundled";

  interface Callback {
    void onComplete(@NonNull Map<String, Object> status);

    void onError(@NonNull String error);
  }

  private PlayServicesRuntime() {}

  static void initialize(
      @NonNull Context context,
      boolean enableGpu,
      boolean enableAutomaticDownload,
      long timeoutMillis,
      @NonNull Callback callback) {
    Map<String, Object> status = new HashMap<>();
    status.put("mode", MODE);
    status.put("interpreterReady", true);
    status.put("compiledModelReady", true);
    status.put("gpuError", null);
    status.put("interpreterError", null);
    status.put("compiledModelError", null);
    callback.onComplete(status);
  }
}
