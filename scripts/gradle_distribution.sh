#!/usr/bin/env bash
# Downloads the Gradle distribution a gradle-wrapper.properties names into the
# place the wrapper looks for it, with retries, so the wrapper finds it there
# and downloads nothing.
#
# Flutter generates gradlew with the Gradle 2.10 wrapper jar, which downloads
# the distribution once, with no retry. services.gradle.org sends that
# download on to github.com, which answered 504 to it in one CI run
# (2026-11-05) and failed the build. The wrapper keeps the zip in
# <GRADLE_USER_HOME>/wrapper/dists/<name>/<hash>/, where <hash> is the MD5 of
# the distribution URL in base 36, and only downloads when no zip is there.
# The zip is checked against the SHA-256 Gradle publishes next to it.
#
# Usage: gradle_distribution.sh <gradle-wrapper.properties>...
set -euo pipefail

home=${GRADLE_USER_HOME:-$HOME/.gradle}
fetch() {
  curl -fsSL --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
    --retry 5 --retry-delay 5 --retry-all-errors -o "$1" "$2"
}

for properties in "$@"; do
  # The value is escaped for java.util.Properties: https\://services...
  url=$(sed -n 's/^distributionUrl=//p' "$properties" | sed 's/\\//g' | tr -d '\r')
  [ -n "$url" ] || { echo "::error::no distributionUrl in $properties"; exit 1; }
  zip_name=${url##*/}
  hash=$(python3 -I -c '
import hashlib, sys
n = int.from_bytes(hashlib.md5(sys.argv[1].encode()).digest(), "big")
digits = "0123456789abcdefghijklmnopqrstuvwxyz"
out = ""
while n:
    n, r = divmod(n, 36)
    out = digits[r] + out
print(out)' "$url")
  dir="$home/wrapper/dists/${zip_name%.zip}/$hash"
  if [ -f "$dir/$zip_name.ok" ] || [ -f "$dir/$zip_name" ]; then
    echo "$zip_name is already in $dir"
    continue
  fi
  mkdir -p "$dir"
  start=$SECONDS
  fetch "$dir/$zip_name.part" "$url"
  expected=$(curl -fsSL --connect-timeout 30 --retry 5 --retry-delay 5 --retry-all-errors "$url.sha256" | tr -d '[:space:]')
  actual=$(sha256sum "$dir/$zip_name.part" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    rm -f "$dir/$zip_name.part"
    echo "::error::$zip_name has SHA-256 $actual, Gradle publishes $expected"
    exit 1
  fi
  mv "$dir/$zip_name.part" "$dir/$zip_name"
  echo "Downloaded $zip_name to $dir in $((SECONDS - start)) s"
done
