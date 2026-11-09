#!/usr/bin/env bash
# One time-boxed stage of the flex delegate build.
#
# A full monolithic flex build takes longer than one GitHub-hosted job may run
# (360 minutes), so the workflows chain several stage jobs. Each stage builds
# with a shared bazel --disk_cache restored from the previous stage, stops
# bazel before the job limit, and the workflow saves the disk cache for the
# next stage. A stage that finishes the build copies the library to DIST_DIR.
#
# Environment:
#   TF_DIR, LIB_NAME, FLEX_OS (linux|windows), FLEX_VARIANT (hermetic|msvc|clang)
#   OUTPUT_USER_ROOT, DISK_CACHE, REPO_CACHE, DIST_DIR
#   JOB_START (epoch seconds), JOB_LIMIT_MIN, SAVE_RESERVE_MIN
#   GITHUB_OUTPUT (result=done|timeout is written there)
set -uo pipefail

now=$(date +%s)
elapsed_min=$(( (now - JOB_START) / 60 ))
budget_min=$(( JOB_LIMIT_MIN - SAVE_RESERVE_MIN - elapsed_min ))
echo "Elapsed ${elapsed_min} min, bazel budget ${budget_min} min."
if [ "$budget_min" -lt 15 ]; then
  echo "Not enough time left in this job to build; leaving it to the next stage."
  echo "result=timeout" >> "$GITHUB_OUTPUT"
  exit 0
fi

cd "$TF_DIR"

if [ "$FLEX_OS" = "windows" ]; then
  export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
fi

bazel_bin="bazelisk"
command -v bazelisk >/dev/null 2>&1 || bazel_bin="bazel"
startup=(--output_user_root="$OUTPUT_USER_ROOT")

flags=(
  -c opt
  --config=monolithic
  --define=tflite_convert_with_select_tf_ops=true
  --define=with_select_tf_ops=true
  --disk_cache="$DISK_CACHE"
  --repository_cache="$REPO_CACHE"
  --repo_env=HERMETIC_PYTHON_VERSION=3.11
  --local_ram_resources=HOST_RAM*.75
  --verbose_failures
  --curses=no
  --color=no
  --show_progress_rate_limit=120
)
if [ "$FLEX_OS" = "windows" ] && [ "$FLEX_VARIANT" = "clang" ]; then
  # TF's --config=win_clang minus win_clang_base's use_dlls switches, which
  # would split protobuf and absl into separate DLLs. The flex DLL must stay a
  # single self-contained file.
  flags+=(
    --compiler=clang-cl
    --extra_toolchains=@local_config_cc//:cc-toolchain-x64_windows-clang-cl
    --extra_execution_platforms=//tensorflow/tools/toolchains/win:x64_windows-clang-cl
    --linkopt=/FORCE:MULTIPLE
    --host_linkopt=/FORCE:MULTIPLE
  )
elif [ "$FLEX_OS" = "windows" ]; then
  # MSVC compiles C as C89-plus-extensions by default; pthreadpool (TF 2.21)
  # includes <stdatomic.h>, which needs C11 and, in VS 2022, the
  # experimental C11 atomics switch. Protobuf refuses MSVC + Bazel unless
  # protobuf_allow_msvc is set (support ends in protobuf 34).
  flags+=(
    --define=protobuf_allow_msvc=true
    --conlyopt=/std:c17
    --conlyopt=/experimental:c11atomics
    --host_conlyopt=/std:c17
    --host_conlyopt=/experimental:c11atomics
  )
fi
target="//tensorflow/lite/delegates/flex:$LIB_NAME"

echo "$bazel_bin ${startup[*]} build ${flags[*]} $target"
"$bazel_bin" "${startup[@]}" version

# Print disk use every 15 minutes while bazel runs.
( while true; do sleep 900; echo "[disk $(date -u +%H:%M)]"; df -h 2>/dev/null | grep -vE 'tmpfs|overlay' ; done ) &
monitor=$!

timeout -s INT -k 10m "${budget_min}m" \
  "$bazel_bin" "${startup[@]}" build "${flags[@]}" "$target"
rc=$?
kill "$monitor" 2>/dev/null || true
echo "bazel build exit code: $rc"

stop_bazel() {
  timeout 10m "$bazel_bin" "${startup[@]}" shutdown || true
  if [ "$FLEX_OS" = "windows" ]; then
    taskkill //F //IM java.exe 2>/dev/null || true
    taskkill //F //IM bazel.exe 2>/dev/null || true
  else
    pkill -9 -f 'bazel\(' 2>/dev/null || true
  fi
}

if [ "$rc" -eq 0 ]; then
  out="bazel-bin/tensorflow/lite/delegates/flex/$LIB_NAME"
  mkdir -p "$DIST_DIR"
  cp "$out" "$DIST_DIR/$LIB_NAME"
  stripped="no"
  if [ "$FLEX_OS" = "linux" ]; then
    # Like the 2.20.0 file: drop the static symbol table and debug info
    # (about 120 MB); the dynamic symbols the loader uses stay.
    chmod u+w "$DIST_DIR/$LIB_NAME"
    echo "size before strip: $(wc -c < "$DIST_DIR/$LIB_NAME")"
    strip --strip-unneeded "$DIST_DIR/$LIB_NAME"
    stripped="strip --strip-unneeded"
  fi
  ( cd "$DIST_DIR" && sha256sum "$LIB_NAME" > "$LIB_NAME.sha256" )
  {
    echo "library: $LIB_NAME"
    echo "tensorflow: $(git describe --tags --always) $(git rev-parse HEAD)"
    echo "bazel: $("$bazel_bin" "${startup[@]}" version 2>/dev/null | grep -m1 'Build label')"
    echo "stripped: $stripped"
    echo "os: $FLEX_OS, variant: $FLEX_VARIANT"
    echo "runner: ${ImageOS:-unknown} ${ImageVersion:-unknown}"
    echo "command: bazel build ${flags[*]} $target"
    echo "sha256: $(cut -d' ' -f1 "$DIST_DIR/$LIB_NAME.sha256")"
    echo "size: $(wc -c < "$DIST_DIR/$LIB_NAME")"
  } > "$DIST_DIR/build-info.txt"
  cat "$DIST_DIR/build-info.txt"
  stop_bazel
  echo "result=done" >> "$GITHUB_OUTPUT"
  exit 0
fi

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "Stage time budget used up; the next stage continues from the disk cache."
  stop_bazel
  echo "result=timeout" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "bazel failed (exit $rc), not a timeout."
stop_bazel
exit "$rc"
