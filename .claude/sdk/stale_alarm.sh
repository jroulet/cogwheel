#!/usr/bin/env bash
# .claude/sdk/stale_alarm.sh — exit when a logfile stops advancing.
#
# Usage: stale_alarm.sh <log_path> [stale_seconds=900]
#
# Poll the log's mtime every 30 s; exit 0 the moment it is older than
# <stale_seconds>. Launch as a background task next to ANY long-running
# job: the harness notifies on task exit, so this converts "log went
# quiet" (a wedge, an OOM'd child, a crawl) into an active notification
# instead of indistinguishable silence. Companion to watchdog.sh, which
# additionally KILLS SDK builds; this one only alerts.
set -u
LOG="${1:?usage: stale_alarm.sh <log_path> [stale_seconds]}"
STALE="${2:-900}"

# Portable mtime (GNU coreutils, then BSD/macOS).
_stat_mtime() { stat -c '%Y' "$1" 2>/dev/null || stat -f '%m' "$1" 2>/dev/null || echo 0; }

# Wait for the log to appear (up to 5 min), then watch it.
for _ in $(seq 60); do
  [[ -f "$LOG" ]] && break
  sleep 5
done
[[ -f "$LOG" ]] || { echo "stale_alarm: $LOG never appeared"; exit 1; }

while :; do
  now=$(date +%s)
  mt=$(_stat_mtime "$LOG")
  if (( now - mt > STALE )); then
    echo "STALE: $LOG silent for $((now - mt))s (threshold ${STALE}s) — check the job"
    exit 0
  fi
  sleep 30
done
