package com.hugocornellier.flutter_litert;

import android.content.Context;
import android.os.SystemClock;
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
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Initializes the LiteRT runtime provided by Google Play services
 * (flutterLitert.runtime=playServices).
 *
 * <p>The Interpreter runtime (TfLiteNative) and the CompiledModel runtime (LiteRt) initialize
 * independently and in parallel, so a device that can provide only one of them still gets that
 * one. Each outcome is recorded in libflutter_litert_runtime.so, which every Dart isolate reads.
 * A runtime that is ready is not initialized again; a failed one is retried by the next call.
 *
 * <p>Every call reports back within its timeout. A Play services task that has not finished by
 * then is recorded as failed, so a stalled module download cannot block the caller forever.
 */
final class PlayServicesRuntime {
  static final String MODE = "playServices";

  private static final int STATE_READY = 1;
  private static final int STATE_FAILED = 2;

  /** Runs one initialization at a time, and may block. */
  private static final Executor executor = Executors.newSingleThreadExecutor();

  private static final Object lock = new Object();
  private static boolean interpreterReady;
  private static boolean compiledModelReady;

  static {
    System.loadLibrary("flutter_litert_runtime");
  }

  interface Callback {
    void onComplete(@NonNull Map<String, Object> status);

    /** Initialization itself failed unexpectedly; the runtime state is unchanged. */
    void onError(@NonNull String error);
  }

  private PlayServicesRuntime() {}

  static void initialize(
      @NonNull Context context,
      boolean enableGpu,
      boolean enableAutomaticDownload,
      long timeoutMillis,
      @NonNull Callback callback) {
    final Context appContext = context.getApplicationContext();
    executor.execute(
        () -> {
          Map<String, Object> status;
          try {
            status = initializeBlocking(appContext, enableGpu, enableAutomaticDownload, timeoutMillis);
          } catch (Throwable t) {
            callback.onError(t.toString());
            return;
          }
          callback.onComplete(status);
        });
  }

  private static Map<String, Object> initializeBlocking(
      Context context, boolean enableGpu, boolean enableAutomaticDownload, long timeoutMillis) {
    final long deadline = SystemClock.elapsedRealtime() + timeoutMillis;
    final boolean compiledModelWasReady;
    final Task<Void> interpreter;
    Task<CompiledModelNativeHandle> compiledModel;
    synchronized (lock) {
      compiledModelWasReady = compiledModelReady;
      interpreter =
          interpreterReady
              ? Tasks.forResult(null)
              : TfLiteNative.initialize(
                  context,
                  TfLiteInitializationOptions.builder()
                      .setEnableAutomaticDownload(enableAutomaticDownload)
                      .build());
      // Once attached, the CompiledModel runtime is fixed for the process. A
      // later request for GPU support cannot add it, so report it as is.
      compiledModel =
          compiledModelWasReady
              ? Tasks.forResult(null)
              : initializeCompiledModel(context, enableGpu, enableAutomaticDownload);
    }

    Failure interpreterFailure = await(interpreter, deadline, "TfLiteNative.initialize");
    Failure compiledModelFailure =
        await(
            compiledModel,
            deadline,
            enableGpu ? "LiteRt.initializeNative with GPU" : "LiteRt.initializeNative");

    String gpuError = null;
    // The GPU accelerator module is not available on every device, so fall
    // back to the CPU-only runtime when the GPU request is what failed. When
    // the Interpreter runtime failed too, or the request timed out, Play
    // services or its LiteRT module is the problem, and a second request
    // would only fail again and double the time to report it.
    if (enableGpu
        && compiledModelFailure != null
        && !compiledModelFailure.timedOut
        && interpreterFailure == null
        && !compiledModelWasReady) {
      gpuError = compiledModelFailure.message;
      compiledModel = initializeCompiledModel(context, false, enableAutomaticDownload);
      compiledModelFailure = await(compiledModel, deadline, "LiteRt.initializeNative");
    }

    Map<String, Object> status = new HashMap<>();
    status.put("mode", MODE);
    status.put("interpreterError", recordInterpreter(interpreterFailure));
    status.put("compiledModelError", recordCompiledModel(compiledModel, compiledModelFailure));
    status.put("gpuError", gpuError);
    synchronized (lock) {
      status.put("interpreterReady", interpreterReady);
      status.put("compiledModelReady", compiledModelReady);
      // Lets Dart tell whether this call decided the CompiledModel runtime's
      // GPU support, or found it already fixed by an earlier call.
      status.put("compiledModelAttached", compiledModelReady && !compiledModelWasReady);
    }
    return status;
  }

  private static Task<CompiledModelNativeHandle> initializeCompiledModel(
      Context context, boolean enableGpu, boolean enableAutomaticDownload) {
    return LiteRt.initializeNative(
        context,
        LiteRtInitializationOptions.builder()
            .setEnableGpu(enableGpu)
            .setEnableAutomaticDownload(enableAutomaticDownload)
            .build());
  }

  /** Why a task failed, or null when it succeeded. */
  private static final class Failure {
    final String message;
    final boolean timedOut;

    Failure(String message, boolean timedOut) {
      this.message = message;
      this.timedOut = timedOut;
    }
  }

  /** Runs on {@link #executor}, never the main thread, so it may block. */
  @Nullable
  private static Failure await(Task<?> task, long deadline, String step) {
    long remaining = Math.max(0, deadline - SystemClock.elapsedRealtime());
    try {
      Tasks.await(task, remaining, TimeUnit.MILLISECONDS);
      return null;
    } catch (TimeoutException e) {
      return new Failure(
          step
              + " did not finish in time. Google Play services may still be downloading the"
              + " LiteRT module",
          true);
    } catch (ExecutionException e) {
      Throwable cause = e.getCause() == null ? e : e.getCause();
      return new Failure(step + " failed: " + cause, false);
    } catch (CancellationException e) {
      return new Failure(step + " was cancelled", false);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return new Failure(step + " was interrupted", false);
    }
  }

  @Nullable
  private static String recordInterpreter(@Nullable Failure failure) {
    synchronized (lock) {
      if (interpreterReady) return null;
      if (failure == null) {
        interpreterReady = true;
        nativeSetInterpreterState(STATE_READY, null);
        return null;
      }
      nativeSetInterpreterState(STATE_FAILED, failure.message);
      return failure.message;
    }
  }

  @Nullable
  private static String recordCompiledModel(
      Task<CompiledModelNativeHandle> task, @Nullable Failure failure) {
    synchronized (lock) {
      if (compiledModelReady) return null;
      String error;
      if (failure == null) {
        error = nativeAttachCompiledModel(task.getResult());
        if (error == null) {
          compiledModelReady = true;
          return null;
        }
      } else {
        error = failure.message;
        nativeSetCompiledModelFailure(error);
      }
      return error;
    }
  }

  private static native void nativeSetInterpreterState(int state, @Nullable String error);

  @Nullable
  private static native String nativeAttachCompiledModel(CompiledModelNativeHandle handle);

  private static native void nativeSetCompiledModelFailure(@Nullable String error);
}
