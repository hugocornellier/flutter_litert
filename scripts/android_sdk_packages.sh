#!/usr/bin/env bash
# Installs Android SDK packages with sdkmanager, each with up to three tries.
#
# reactivecircus/android-emulator-runner downloads the emulator and the system
# image itself, once each, and a download from Google that arrives broken
# ("Error on ZipFile unknown archive", "Connection reset") fails the job before
# any test runs. Seen in 2 jobs. Run before that step, this leaves the packages
# installed, so the step has nothing left to download.
#
# Usage: android_sdk_packages.sh <package>...
#   e.g. android_sdk_packages.sh emulator 'system-images;android-34;default;x86_64'
set -u

sdk=${ANDROID_HOME:?ANDROID_HOME is not set}
sdkmanager="$sdk/cmdline-tools/latest/bin/sdkmanager"

yes | "$sdkmanager" --licenses > /dev/null

for package in "$@"; do
  # A package a;b;c is installed in $ANDROID_HOME/a/b/c.
  dir="$sdk/${package//;//}"
  try=1
  # sdkmanager names the cause on stderr, which stays in the job's log.
  until "$sdkmanager" --install "$package" --channel=0 > /dev/null && [ -e "$dir/package.xml" ]; do
    if [ "$try" -ge 3 ]; then
      echo "::error::the Android SDK package $package could not be downloaded in $try tries"
      exit 1
    fi
    echo "::warning::the download of the Android SDK package $package broke off (try $try), downloading it again"
    # What a broken download leaves behind, as for the NDK in
    # android_flutter_test.sh.
    rm -rf "$dir" "$sdk"/.temp
    try=$((try + 1))
    sleep 10
  done
  echo "Android SDK package $package is installed"
done
