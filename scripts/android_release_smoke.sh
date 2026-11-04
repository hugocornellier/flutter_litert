#!/usr/bin/env bash
# Installs the Play services host's release smoke APK on a device, launches it,
# and waits for the result line lib/release_smoke.dart prints to logcat.
#
#   scripts/android_release_smoke.sh app-release.apk [serial] [timeout seconds]
#
# Exits 0 on PASS, 1 on FAIL or when no result appears in time.
set -u

apk=$1
serial=${2:-emulator-5554}
limit=${3:-600}
app=com.hugocornellier.flutter_litert_play_services_test_host
marker='flutter_litert release smoke: '

adb -s "$serial" install -r "$apk" || exit 1
adb -s "$serial" logcat -c
adb -s "$serial" shell am start -W -n "$app/.MainActivity" || exit 1

waited=0
result=
while [ "$waited" -lt "$limit" ]; do
  line=$(adb -s "$serial" logcat -d -s flutter:I | grep -F "$marker" | tail -n 1)
  if [ -n "$line" ]; then
    echo "$line"
    case $line in
      *"${marker}PASS"*) exit 0 ;;
    esac
    result=failed
    break
  fi
  if [ -z "$(adb -s "$serial" shell pidof "$app" | tr -d '\r')" ]; then
    result="the app died before printing a result"
    break
  fi
  sleep 5
  waited=$((waited + 5))
done

echo "release smoke: ${result:-no result after ${limit}s}"
# The app's crash, if any, and the LiteRT lines.
adb -s "$serial" logcat -d -s AndroidRuntime:E flutter:I | tail -n 80
adb -s "$serial" logcat -d | grep -E "LiteRt|litert|TfLite|tflite" | tail -n 80
exit 1
