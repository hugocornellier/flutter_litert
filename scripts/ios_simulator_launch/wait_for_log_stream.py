#!/usr/bin/env python3
"""Holds an iOS simulator app launch until flutter's log stream is connected.

`flutter test -d <simulator>` finds the app's Dart VM service only through
the "Dart VM service is listening" line, read from a `simctl spawn <udid>
log stream` that it starts at the same moment as `simctl launch`. `log
stream` shows only lines logged after it connects, and flutter waits for the
line with no timeout. On a busy runner the simulator can take seconds (15 s
for the first `log` after a boot) to start that stream, so an app that
prints the line first leaves `flutter test` waiting forever.

The xcrun stand-in next to this file runs this just before `simctl launch`.
It finds the stream started by the same flutter process, waits until the
simulator's diagnosticd reports that stream as connected, and then lets the
launch go ahead. It never fails the launch: if the stream is not seen in
time it logs a warning and returns, which is the behaviour without it.

Usage: wait_for_log_stream.py <simulator udid> <flutter_tools pid>
Messages go to the file named by IOS_LAUNCH_GATE_LOG (stderr if unset).
"""

import datetime
import os
import subprocess
import sys
import time

TIMEOUT_S = 120
# How flutter_tools' stream shows up as a process inside the simulator.
STREAM_ARGS = 'log stream --style json --predicate eventType = logEvent'


def log(message):
    stamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f'{stamp} ios launch gate: {message}\n'
    path = os.environ.get('IOS_LAUNCH_GATE_LOG')
    if path:
        with open(path, 'a') as f:
            f.write(line)
    else:
        sys.stderr.write(line)


def processes():
    """Yields (pid, ppid, start time, args) for every process on the host.

    Simulator processes are host processes, so this sees both flutter's
    `simctl spawn` and the `log` process it started inside the simulator.
    """
    out = subprocess.run(
        ['ps', '-axww', '-o', 'pid=,ppid=,lstart=,args='],
        capture_output=True, text=True,
        env=dict(os.environ, LC_ALL='C')).stdout
    for line in out.splitlines():
        # pid ppid Tue Nov 10 05:12:45 2026 args...
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        try:
            started = time.mktime(
                time.strptime(' '.join(parts[2:7]), '%a %b %d %H:%M:%S %Y'))
            yield int(parts[0]), int(parts[1]), started, parts[7]
        except ValueError:
            continue


def connected_at(udid, pid):
    """When diagnosticd accepted the stream of `log` process `pid`, or None."""
    out = subprocess.run(
        ['/usr/bin/xcrun', 'simctl', 'spawn', udid, 'log', 'show',
         '--last', '5m', '--style', 'compact', '--predicate',
         'process == "diagnosticd" AND eventMessage BEGINSWITH '
         f'"System mode client started - log ({pid})"'],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'System mode client started' in line:
            return line[:23]
    return None


def main():
    udid, flutter_pid = sys.argv[1], int(sys.argv[2])
    begin = time.monotonic()
    while time.monotonic() - begin < TIMEOUT_S:
        procs = list(processes())
        hosts = [p for p in procs
                 if p[1] == flutter_pid and f'simctl spawn {udid} log stream' in p[3]]
        if hosts:
            host_start = max(p[2] for p in hosts)
            # Leftover streams from earlier flutter runs started before this one.
            streams = [p for p in procs
                       if p[3].startswith(STREAM_ARGS) and p[2] >= host_start]
            if streams:
                pid = max(streams, key=lambda p: (p[2], p[0]))[0]
                at = connected_at(udid, pid)
                if at:
                    log(f'flutter log stream (simulator pid {pid}) connected at '
                        f'{at}, {time.monotonic() - begin:.1f}s after the launch '
                        'request; launching')
                    # The app cannot print before it starts, so this second is
                    # the smallest possible margin.
                    time.sleep(1)
                    return 0
        time.sleep(0.5)
    log(f'warning: flutter log stream not connected after {TIMEOUT_S}s; '
        'launching anyway')
    return 0


if __name__ == '__main__':
    sys.exit(main())
