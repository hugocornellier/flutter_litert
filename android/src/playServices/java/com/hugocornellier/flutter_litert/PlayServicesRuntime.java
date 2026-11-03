package com.hugocornellier.flutter_litert;

import android.content.Context;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.google.android.gms.litert.LiteRt;
import com.google.android.gms.litert.client.CompiledModelNativeHandle;
import com.google.android.gms.litert.client.LiteRtInitializationOptions;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.java.TfLiteNative;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Initializes the LiteRT runtime provided by Google Play services
 * (flutterLitert.runtime=playServices).
 *
 * <p>The Interpreter runtime (TfLiteNative) and the CompiledModel runtime (LiteRt) initialize
 * independently and in parallel, so a device that can provide only one of them still gets that
 * one. Each outcome is recorded in libflutter_litert_runtime.so, which every Dart isolate reads.
 * A runtime that is ready is not initialized again; a failed one is retried by the next call.
 */
final class PlayServicesRuntime {
  static final String MODE = "playServices";

  private static final int STATE_READY = 1;
  private static final int STATE_FAILED = 2;

  private static final Executor executor = Executors.newSingleThreadExecutor();
  private static final Object lock = new Object();
  private static boolean interpreterReady;
  private static boolean compiledModelReady;

  static {
    System.loadLibrary("flutter_litert_runtime");
  }

  interface Callback {
    void onComplete(@NonNull Map<String, Object> status);
  }

  private PlayServicesRuntime() {}

  static void initialize(
      @NonNull Context context,
      boolean enableGpu,
      boolean enableAutomaticDownload,
      @NonNull Callback callback) {
    final Context appContext = context.getApplicationContext();
    final Task<Void> interpreter;
    final Task<CompiledModelNativeHandle> compiledModel;
    synchronized (lock) {
      interpreter =
          interpreterReady
              ? Tasks.forResult(null)
              : TfLiteNative.initialize(
                  appContext,
                  TfLiteInitializationOptions.builder()
                      .setEnableAutomaticDownload(enableAutomaticDownload)
                      .build());
      // Once attached, the CompiledModel runtime is fixed for the process. A
      // later request for GPU support cannot add it, so report it as is.
      compiledModel =
          compiledModelReady
              ? Tasks.forResult(null)
              : LiteRt.initializeNative(
                  appContext,
                  LiteRtInitializationOptions.builder()
                      .setEnableGpu(enableGpu)
                      .setEnableAutomaticDownload(enableAutomaticDownload)
                      .build());
    }

    Tasks.whenAllComplete(interpreter, compiledModel)
        .addOnCompleteListener(
            executor,
            ignored -> {
              Map<String, Object> status = new HashMap<>();
              status.put("mode", MODE);
              status.put("interpreterError", recordInterpreter(interpreter));
              Task<CompiledModelNativeHandle> result = compiledModel;
              String gpuError = null;
              if (enableGpu && !result.isSuccessful() && !isCompiledModelReady()) {
                // The GPU accelerator module is not available on every device.
                // Fall back to the CPU-only runtime rather than failing.
                gpuError = describe("LiteRt.initializeNative with GPU", result);
                result = initializeCpuOnly(appContext, enableAutomaticDownload);
              }
              status.put("gpuError", gpuError);
              status.put("compiledModelError", recordCompiledModel(result));
              synchronized (lock) {
                status.put("interpreterReady", interpreterReady);
                status.put("compiledModelReady", compiledModelReady);
              }
              callback.onComplete(status);
            });
  }

  private static boolean isCompiledModelReady() {
    synchronized (lock) {
      return compiledModelReady;
    }
  }

  /** Runs on {@link #executor}, never the main thread, so it may block. */
  private static Task<CompiledModelNativeHandle> initializeCpuOnly(
      Context context, boolean enableAutomaticDownload) {
    Task<CompiledModelNativeHandle> task =
        LiteRt.initializeNative(
            context,
            LiteRtInitializationOptions.builder()
                .setEnableGpu(false)
                .setEnableAutomaticDownload(enableAutomaticDownload)
                .build());
    try {
      Tasks.await(task);
    } catch (Exception ignored) {
      // The task records the failure; recordCompiledModel reports it.
    }
    return task;
  }

  @Nullable
  private static String recordInterpreter(Task<Void> task) {
    synchronized (lock) {
      if (interpreterReady) return null;
      if (task.isSuccessful()) {
        interpreterReady = true;
        nativeSetInterpreterState(STATE_READY, null);
        return null;
      }
      String error = describe("TfLiteNative.initialize", task);
      nativeSetInterpreterState(STATE_FAILED, error);
      return error;
    }
  }

  @Nullable
  private static String recordCompiledModel(Task<CompiledModelNativeHandle> task) {
    synchronized (lock) {
      if (compiledModelReady) return null;
      String error;
      if (task.isSuccessful()) {
        error = nativeAttachCompiledModel(task.getResult());
        if (error == null) {
          compiledModelReady = true;
          return null;
        }
      } else {
        error = describe("LiteRt.initializeNative", task);
        nativeSetCompiledModelFailure(error);
      }
      return error;
    }
  }

  private static String describe(String step, Task<?> task) {
    if (task.isCanceled()) return step + " was cancelled";
    Exception exception = task.getException();
    return step + " failed: " + (exception == null ? "unknown error" : exception.toString());
  }

  private static native void nativeSetInterpreterState(int state, @Nullable String error);

  @Nullable
  private static native String nativeAttachCompiledModel(CompiledModelNativeHandle handle);

  private static native void nativeSetCompiledModelFailure(@Nullable String error);
}
