#!/usr/bin/env bash
# .claude/sdk/build_monitor.sh — the driver's build progress stream.
#
# Usage: build_monitor.sh <log_path> [poll_seconds=120]
#        (arm it as the `command` of a persistent Monitor)
#
# Each stdout line becomes one driver notification. Exits 0 at terminal, so a
# finished build stops its own watcher.
#
# WHY THIS EXISTS. The driver used to hand-roll this pipeline inline at every
# launch, and got it wrong three different ways — each cost real build time:
#
#   1. SELF-MATCH. The Monitor's own command text is echoed into the build log
#      by the harness, and it contains every marker the filter greps for. An
#      unfiltered watcher matches itself on the first poll and reports a stale
#      build's markers forever. Fixed by dropping `Monitor(persistent` lines.
#   2. DEDUPE ON REPEAT (measured 2026-08-12). Comparing the newest matching
#      line against the last emitted one is blind to a byte-identical repeat.
#      `[file-based]` lines carry NO timestamp, so the second
#      "Plan written to <dir>/plan.json" after a rejected-and-resubmitted plan
#      is identical to the first. A monitor stayed silent through it and the
#      build waited 8 min on driver approval with the watchdog clock running.
#      Fixed by tracking the COUNT of matching lines and emitting those past
#      the previous count.
#   3. NO TEARDOWN. Killing or relaunching a build does not stop its monitor,
#      so a watcher sits on a dead log reporting stalls that mean nothing —
#      and a watcher on a dead log can never reach its terminal condition, so
#      it is permanently indistinguishable from a hung build. Four accumulated
#      in one session (2026-08-06). Fixed by exiting when no build process
#      remains, which needs no cooperation from whoever does the killing.
#
# Health is EITHER the log advancing OR the build process being alive: the
# build log legitimately freezes while the tree gate writes to its own log.
set -u

LOG="${1:?usage: build_monitor.sh <log_path> [poll_seconds] [build_pid]}"
POLL="${2:-120}"
# Optional: the PID this log belongs to. STRONGLY preferred — without it the
# liveness check matches ANY sdk/build.py, so a second, unrelated build keeps
# this monitor alive at a dead log, which is the very failure it exists to
# prevent. launch_build.sh knows the PID and prints it in the armed command.
BUILD_PID="${3:-}"

# Markers worth one driver invocation each.
FILTER='Phase [0-9]:|Triage:|plan_ready|Plan written|Waiting for a decision|Waiting for approval|Coder checkpoint|Inspector: (PASS|found issues)|Professor: (PASS|CONCERN|FAIL)|GATE FAILURE|ESCALATION|escalation|Traceback|TimeoutError|KILLED|Build (failed|complete)'
# Terminal markers: seeing one means stop watching.
TERMINAL='^(\[[^]]*\])?[[:space:]]*(===[[:space:]]*)?(BUILD REPORT|Build failed|Build cancelled|KILLED BY WATCHDOG|GATE FAILURE|BUILD STRANDED)'
# The build process pattern. Bracket idiom so the check cannot match itself.
BUILD_PAT='sdk/[b]uild.py'

# Strip the harness's echo of a Monitor command (blind spot 1).
_matches() { grep -av 'Monitor(persistent' "$LOG" 2>/dev/null | grep -aE "$FILTER"; }
_count()   { _matches | grep -ac '' 2>/dev/null || echo 0; }

for _ in $(seq 30); do [ -f "$LOG" ] && break; sleep 10; done
if [ ! -f "$LOG" ]; then
  echo "build_monitor: $LOG never appeared (5 min) — launcher failed"; exit 0
fi

seen="$(_count)"
if [ -n "$BUILD_PID" ]; then
  echo "build_monitor: watching $(basename "$LOG") (baseline $seen markers, pid $BUILD_PID)"
else
  # Say it out loud. A silent degradation to the coarse check is the same bug
  # class this script exists to close: any other running build would keep this
  # monitor alive at a dead log.
  echo "build_monitor: watching $(basename "$LOG") (baseline $seen markers) — WARNING: no build_pid given, liveness falls back to matching ANY sdk/build.py; pass the PID (launch_build.sh prints it)"
fi
stall=0

while :; do
  cur="$(_count)"
  if [ "$cur" -gt "$seen" ]; then
    # Emit every NEW occurrence, not just the newest line (blind spot 2).
    # Cap the burst so a chatty stretch cannot trip the harness rate limit.
    new=$((cur - seen))
    if [ "$new" -gt 4 ]; then
      echo "(+$new markers; newest 4)"
    fi
    _matches | tail -n "$new" | tail -4
    seen="$cur"
    stall=0
  else
    stall=$((stall + 1))
    # Announce a stall ONCE on entering it, then stay quiet.
    if [ "$stall" -eq 6 ]; then
      echo "STALL: no new marker in ~$((6 * POLL / 60)) min (count $seen) — investigate, do not wait"
    fi
  fi

  if grep -av 'Monitor(persistent' "$LOG" 2>/dev/null | grep -qaE "$TERMINAL"; then
    grep -av 'Monitor(persistent' "$LOG" | grep -aE "$TERMINAL" | tail -2
    exit 0
  fi

  # Teardown (blind spot 3): once the build is gone nothing will ever advance
  # this log again. Exit rather than report stalls at a corpse.
  if [ -n "$BUILD_PID" ]; then
    _alive() { kill -0 "$BUILD_PID" 2>/dev/null; }
  else
    _alive() { pgrep -f "$BUILD_PAT" >/dev/null 2>&1; }
  fi
  if ! _alive; then
    echo "build_monitor: build gone${BUILD_PID:+ (pid $BUILD_PID)} (count $seen); last lines:"
    tail -3 "$LOG"
    exit 0
  fi

  sleep "$POLL"
done
