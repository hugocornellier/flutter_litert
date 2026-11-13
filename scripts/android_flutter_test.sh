#!/usr/bin/env bash
# Runs one integration test file with `flutter test` on an Android emulator,
# and ends an attempt that installs the app but never starts a test.
#
#   scripts/android_flutter_test.sh integration_test/x_test.dart [flutter test args]
#
# Run it from the app's directory. `flutter test --timeout` is enforced inside
# the app, so it cannot end a run whose app never comes up. Seen in CI:
#   - API 24: `adb install` never returns (4 of 17 jobs on main, three of them
#     silent for 6 hours). adb has sent the whole APK and the package manager
#     still waits to read more of it. Another install in the same emulator
#     usually goes through, now and then only after Android is restarted.
#   - Play Store image: the app starts, Android restarts its activity, and
#     flutter test waits without end for the one it started.
#   - flutter does not find the device although adb answers, and gives up in
#     2 seconds (once in about 95 jobs).
# Here every attempt has a hard limit, and a watchdog ends an attempt that
# prints no test line soon after the APK is built. By default nothing is tried
# again, except a build whose NDK download broke off: a launch failure, a
# device that was not found and a failed test all fail the script. The push and
# PR workflow (flutter-ci.yml) runs with these defaults, so a flaky test shows
# up red instead of passing on a second attempt. ANDROID_TEST_LAUNCH_ATTEMPTS
# and ANDROID_TEST_ATTEMPTS allow more attempts; only device-tests-manual.yml
# sets them.
#
# Environment:
#   ANDROID_TEST_SERIAL           device, default emulator-5554
#   ANDROID_TEST_LAUNCH_LIMIT     seconds from the built APK to the first test
#                                 line, default 180 (a healthy launch takes
#                                 under 20)
#   ANDROID_TEST_ATTEMPT_LIMIT    seconds for one whole attempt, default 1500
#   ANDROID_TEST_LAUNCH_ATTEMPTS  attempts while no test has started or flutter
#                                 did not find the device, default 1
#   ANDROID_TEST_ATTEMPTS         attempts for a failure after a test started,
#                                 default 1
#   ANDROID_TEST_INSTALL_FIRST    an APK of the same app to install before
#                                 flutter test when the app is not on the
#                                 device, so that flutter's install is an
#                                 update; default none
#   ANDROID_TEST_LOG_DIR          default $GITHUB_WORKSPACE/build/android-test-logs
#
# Exits 0 when the tests pass, 1 otherwise. Each attempt's output and logcat
# stay in the log directory, with the device state at the moment the watchdog
# fired; the file retried there tells the workflow to upload them even when
# another attempt passed.
set -u

test_file=$1
shift
flutter_args=("$@")

serial=${ANDROID_TEST_SERIAL:-emulator-5554}
launch_limit=${ANDROID_TEST_LAUNCH_LIMIT:-180}
attempt_limit=${ANDROID_TEST_ATTEMPT_LIMIT:-1500}
launch_attempts=${ANDROID_TEST_LAUNCH_ATTEMPTS:-1}
test_attempts=${ANDROID_TEST_ATTEMPTS:-1}
install_first=${ANDROID_TEST_INSTALL_FIRST:-}
log_dir=${ANDROID_TEST_LOG_DIR:-${GITHUB_WORKSPACE:-$PWD}/build/android-test-logs}
app=$(sed -n 's/.*applicationId *= *"\([^"]*\)".*/\1/p' android/app/build.gradle* 2> /dev/null | head -n 1)
name=$(basename "$test_file" .dart)
mkdir -p "$log_dir"

run_pid=
tail_pid=
logcat_pid=
stop_background() {
  kill ${run_pid:+"$run_pid"} ${tail_pid:+"$tail_pid"} ${logcat_pid:+"$logcat_pid"} 2> /dev/null
}
trap stop_background EXIT

# Every adb call is bounded: adb is one of the things that stops answering.
device() {
  local limit=$1
  shift
  timeout -k 5 "$limit" adb -s "$serial" "$@"
}

built() {
  grep -aq 'Built build/' "$1"
}

# The expanded reporter prints a line when a test starts. "loading ..." lines
# name no test, and a launch that dies ends on "+0 -1: Some tests failed." with
# no test behind it, so neither counts. Not grep -q at the end: it can SIGPIPE
# the first grep.
test_started() {
  grep -aE '[0-9]+:[0-9]{2} \+[0-9]+' "$1" |
    grep -v -e ': loading ' -e ': Some tests failed' > /dev/null
}

# What the host and the device are doing while the launch is stuck. Thread
# dumps need root, which the Play Store image does not give; they are skipped
# there.
dump_state() {
  local out=$1 pid
  {
    echo "== host: CPU, then emulator, adb and flutter processes"
    grep -m 1 'model name' /proc/cpuinfo
    ps -eo pid,ppid,pcpu,etimes,args | grep -E '[q]emu-system|[a]db[^ ]* |[f]lutter_tools' | cut -c 1-300
    # wchar shows how much of the APK a stuck `adb install` has sent.
    for pid in $(pgrep -f "adb[^ ]* -s $serial install"); do
      echo "adb install $pid: $(tr '\n' ' ' < "/proc/$pid/io")"
    done
    echo "== adb devices"
    timeout -k 5 20 adb devices -l
    if ! device 20 shell true; then
      echo "== adb shell does not answer on $serial"
      return
    fi
    echo "== device: boot, uptime, memory, storage"
    device 20 shell 'getprop sys.boot_completed; uptime; head -n 3 /proc/meminfo; df /data'
    echo "== device: package"
    device 30 shell pm path "$app"
    echo "== device: processes"
    device 30 shell 'if [ "$(ps -A 2> /dev/null | wc -l)" -gt 5 ]; then ps -A; else ps; fi'
    echo "== device: busiest processes"
    device 30 shell top -n 1 -m 15
    echo "== device: install sessions and what they have received"
    device 30 shell dumpsys package installs | head -n 60
    echo "== device: top activity"
    device 30 shell dumpsys activity activities | grep -E 'mResumedActivity|mFocusedActivity|ResumedActivity|Run #' | head -n 20
    for process in adbd system_server "$app"; do
      pid=$(device 20 shell pidof "$process" | tr -d '\r')
      echo "== device: threads of $process (pid ${pid:-none})"
      [ -n "$pid" ] || continue
      # Android 7's ps lists threads with -t, later ones with -T.
      device 30 shell "ps -t $pid 2> /dev/null; ps -T -p $pid 2> /dev/null"
      device 60 shell su 0 debuggerd -b "$pid" | head -n 2000
      # ART writes the Java stacks under /data/anr on SIGQUIT.
      [ "$process" = adbd ] || device 20 shell su 0 kill -3 "$pid"
    done
    sleep 5
    echo "== device: Java stacks"
    device 60 shell 'su 0 sh -c "cat /data/anr/*"' | head -n 6000
  } > "$out" 2>&1
}

# On the Play Store image Android can handle the broadcast of a new app's
# install seconds late and then restarts the activity under the running test.
# With the app already on the device, and that broadcast handled, flutter
# test's own install is an update. A failure here only warns.
install_app_first() {
  local started=$SECONDS
  [ -n "$install_first" ] || return 0
  [ -z "$(device 30 shell pm path "$app" 2> /dev/null)" ] || return 0
  if device 300 install -t -r "$install_first" > /dev/null 2>&1 &&
    device 300 shell am wait-for-broadcast-idle > /dev/null 2>&1; then
    echo "$name attempt $attempt: installed $install_first first ($((SECONDS - started))s)"
  else
    echo "::warning::$name attempt $attempt: could not install $install_first first, Android may restart the app under the test"
  fi
}

# Sets rc and killed. Output goes to a file, not a pipe into tee, so a stray
# child holding the pipe cannot block the script.
run_attempt() {
  local log=$1 built_at='' first=''
  killed=0
  install_app_first
  device 20 logcat -c > /dev/null 2>&1
  adb -s "$serial" logcat -v threadtime > "${log%.log}.logcat.txt" 2>&1 &
  logcat_pid=$!
  echo "device: $(device 20 shell 'head -n 1 /proc/meminfo; df /data | tail -n 1' 2>&1 | tr -s ' \r\n' ' ')"
  : > "$log"
  # timeout runs flutter in its own process group and signals the whole
  # group, so a stuck adb child ends with it.
  timeout -k 60 "$attempt_limit" flutter test "$test_file" -d "$serial" \
    --reporter expanded "${flutter_args[@]}" > "$log" 2>&1 &
  run_pid=$!
  tail -n +1 -f --pid="$run_pid" "$log" &
  tail_pid=$!
  while kill -0 "$run_pid" 2> /dev/null; do
    if test_started "$log"; then
      first=$((SECONDS - ${built_at:-$SECONDS}))
      break
    fi
    if [ -z "$built_at" ] && built "$log"; then
      built_at=$SECONDS
    fi
    if [ -n "$built_at" ] && [ $((SECONDS - built_at)) -ge "$launch_limit" ]; then
      killed=1
      # flutter leaves its "Installing ..." line open until the install
      # returns, and a workflow command must start a line.
      echo
      echo "::warning::$name attempt $attempt: no test started within ${launch_limit}s of the built APK, ending the attempt"
      dump_state "${log%.log}.state.txt"
      kill -TERM "$run_pid" 2> /dev/null
      break
    fi
    sleep 2
  done
  rc=0
  wait "$run_pid" || rc=$?
  wait "$tail_pid" 2> /dev/null
  # Flutter may handle TERM and exit 0. A run the watchdog ended never passes.
  if [ "$killed" -eq 1 ] && [ "$rc" -eq 0 ]; then rc=143; fi
  kill "$logcat_pid" 2> /dev/null
  wait "$logcat_pid" 2> /dev/null
  run_pid=
  tail_pid=
  logcat_pid=
  echo "$name attempt $attempt: rc=$rc${first:+, first test line ${first}s after the built APK}"
  if [ "$rc" -ne 0 ]; then
    echo "::group::logcat of $name attempt $attempt"
    grep -aE 'flutter|LiteRt|litert|TfLite|tflite|Dynamite|Chimera|AndroidRuntime|FATAL' "${log%.log}.logcat.txt" | tail -n 300
    echo "::endgroup::"
  fi
}

wait_for_device() {
  local deadline=$((SECONDS + 300))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ "$(device 20 shell getprop sys.boot_completed 2> /dev/null | tr -d '\r')" = 1 ] &&
      device 30 shell pm path android > /dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "::warning::$serial did not answer within 5 minutes"
  return 1
}

# Before another launch in the same emulator. The first retry only clears what
# the ended attempt left; the second also restarts Android, in case the first
# showed that the package manager or adbd stays stuck.
recover() {
  pkill -f "adb[^ ]* -s $serial install" 2> /dev/null
  if [ "$launch_failures" -ge 2 ]; then
    echo "Restarting Android on $serial before attempt $((attempt + 1))"
    device 60 reboot
    sleep 10
  elif device 30 shell pm path android > /dev/null 2>&1; then
    device 30 shell am force-stop "$app"
    return
  else
    echo "adb or the package manager does not answer on $serial, restarting the adb server"
    timeout -k 5 30 adb kill-server
    timeout -k 5 30 adb start-server
  fi
  # The restarted device shows its lock screen again.
  wait_for_device && device 20 shell input keyevent 82
}

attempt=0
build_failures=0
device_failures=0
launch_failures=0
test_failures=0
while :; do
  attempt=$((attempt + 1))
  log="$log_dir/$name.attempt$attempt.log"
  run_attempt "$log"
  [ "$rc" -eq 0 ] && break
  if ! built "$log" && grep -aq 'No supported devices found' "$log"; then
    # flutter asks adb for the device's properties before it builds, and one
    # failed answer makes it report no device.
    device_failures=$((device_failures + 1))
    : > "$log_dir/retried"
    if [ "$device_failures" -ge "$launch_attempts" ]; then
      echo "::error::$name: flutter did not find $serial on $device_failures attempts (rc=$rc)"
      break
    fi
    echo "::warning::$name attempt $attempt: flutter did not find $serial (rc=$rc), trying again once adb answers"
    wait_for_device
    continue
  fi
  if ! built "$log"; then
    # Gradle's SDK-component install can receive a corrupted NDK archive
    # ("Error on ZipFile unknown archive"), which fails assembleDebug. Only
    # then the partial NDK is purged and the build runs again.
    build_failures=$((build_failures + 1))
    if grep -aq 'preparing SDK package NDK' "$log" && [ "$build_failures" -lt 3 ]; then
      echo "::warning::$name attempt $attempt: the NDK download broke off (rc=$rc), building again"
      rm -rf "$ANDROID_HOME"/ndk/* "$ANDROID_HOME"/.temp
      continue
    fi
    echo "::error::$name: the build failed (rc=$rc)"
    break
  fi
  if ! test_started "$log"; then
    launch_failures=$((launch_failures + 1))
    : > "$log_dir/retried"
    if [ "$launch_failures" -ge "$launch_attempts" ]; then
      echo "::error::$name: the app failed to launch on all $launch_failures attempts (rc=$rc)"
      break
    fi
    echo "::warning::$name attempt $attempt: the app was built but no test started (rc=$rc), launching again in the same emulator"
    recover
    continue
  fi
  test_failures=$((test_failures + 1))
  if [ "$test_failures" -ge "$test_attempts" ]; then
    if [ "$rc" -eq 124 ]; then
      echo "::error::$name: timed out after ${attempt_limit}s with tests running"
    else
      echo "::error::$name: tests failed (rc=$rc)"
    fi
    break
  fi
  : > "$log_dir/retried"
  echo "::warning::$name attempt $attempt failed after its tests started (rc=$rc), running it again"
done

[ "$rc" -eq 0 ] && [ "$attempt" -gt 1 ] && echo "$name passed on attempt $attempt"
[ "$rc" -eq 0 ]
