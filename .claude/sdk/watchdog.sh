#!/usr/bin/env bash
# .claude/sdk/watchdog.sh — kill a wedged SDK build if its log goes stale.
#
# Usage:
#   watchdog.sh <log_path> [stale_seconds] [orch_pid]
#
# Polls <log_path> every 30s. If the file's mtime hasn't advanced in
# <stale_seconds> (default 1200 = 20 min), kills the orchestrator subtree.
# Self-exits cleanly when the orchestrator exits naturally.
#
# <orch_pid> is passed by launch_build.sh ($! of the build process). Omit it
# only for hand launches; the pattern fallback then guards the NEWEST
# orchestrator, which is the wrong one whenever two builds overlap.
#
# Pass stale_seconds=0 to disable the staleness check while still
# tracking the orchestrator process.
#
# Exit codes:
#   0 = orchestrator exited normally
#   1 = setup error
#   2 = watchdog killed the build

set -u

LOG_PATH="${1:?usage: watchdog.sh <log_path> [stale_seconds] [orch_pid]}"
STALE_SECONDS="${2:-1200}"  # 600 killed a healthy Opus planning turn (2026-07-10)
ORCH_PID="${3:-}"
POLL_INTERVAL="${WATCHDOG_POLL_INTERVAL:-30}"   # overridable so the probe is fast
STARTUP_GRACE="${WATCHDOG_STARTUP_GRACE:-60}"

WATCHDOG_LOG="${LOG_PATH%.log}.watchdog.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"
}

if [ "$STALE_SECONDS" -le 0 ]; then
    log "Watchdog starting: log=$LOG_PATH stale_threshold=DISABLED poll=${POLL_INTERVAL}s"
else
    log "Watchdog starting: log=$LOG_PATH stale_threshold=${STALE_SECONDS}s poll=${POLL_INTERVAL}s"
fi

# Wait for the log file to appear
for _ in $(seq 1 "$STARTUP_GRACE"); do
    [ -f "$LOG_PATH" ] && break
    sleep 1
done
if [ ! -f "$LOG_PATH" ]; then
    log "ERROR: Log file did not appear within ${STARTUP_GRACE}s. Exiting."
    exit 1
fi

# Find the orchestrator PID.
#
# The entrypoint is build.py, a shim that calls sdk.cli.main() IN-PROCESS, so
# argv reads ".../.claude/sdk/build.py build" and never "cli.py". Matching the
# wrong name here disarmed the watchdog on every launcher-started build from
# 2026-07-27 to 2026-07-30 (F055) while launch_build.sh still printed
# "(watchdog 1200s)". Prefer the PID the launcher already knows.
# PRIMARY KEY: the log path. It is unique per build and already on the
# orchestrator's own command line, so the match is exact, race-free, and
# indifferent to HOW the interpreter was invoked. (Credit: the gw pipeline,
# which must use it — its launcher goes through `conda run`, so $! there is
# the conda wrapper, which "survives a subtree kill and reads as alive".)
BY_LOG=$(pgrep -f -- "sdk/(build|cli)\.py build.*${LOG_PATH}" 2>/dev/null |
         head -1)

if [ -n "$BY_LOG" ]; then
    if [ -n "$ORCH_PID" ] && [ "$ORCH_PID" != "$BY_LOG" ]; then
        # Not fatal, but never silent: this is the signature of a launcher
        # that backgrounds a WRAPPER (conda run, a shell -c) instead of the
        # interpreter. Killing the wrapper leaves the build running.
        log "NOTE: launcher PID $ORCH_PID != log-matched $BY_LOG — the passed"
        log "      PID is probably a wrapper; trusting the log match."
    fi
    ORCH_PID="$BY_LOG"
elif [ -n "$ORCH_PID" ]; then
    log "NOTE: no process names this log yet; using launcher PID $ORCH_PID"
else
    ORCH_PID=$(pgrep -nf '\.claude/sdk/(build|cli)\.py build' 2>/dev/null || true)
    if [ -z "$ORCH_PID" ]; then
        log "ERROR: Could not find orchestrator process. Exiting."
        exit 1
    fi
    log "WARNING: no PID and no log match; took the NEWEST orchestrator by"
    log "         pattern — wrong build if two are running."
fi

if ! kill -0 "$ORCH_PID" 2>/dev/null; then
    log "ERROR: orchestrator PID $ORCH_PID is not alive. Exiting."
    exit 1
fi
log "Watching orchestrator PID $ORCH_PID"

# Portable stat for mtime
_stat_mtime() { stat -c '%Y' "$1" 2>/dev/null || stat -f '%m' "$1" 2>/dev/null || echo 0; }
LAST_MTIME=$(_stat_mtime "$LOG_PATH")
LAST_GROWTH_TIME=$(date +%s)

while true; do
    sleep "$POLL_INTERVAL"

    if ! kill -0 "$ORCH_PID" 2>/dev/null; then
        log "Orchestrator PID $ORCH_PID is gone. Build finished. Watchdog done."
        exit 0
    fi

    CURRENT_MTIME=$(_stat_mtime "$LOG_PATH")
    NOW=$(date +%s)

    if [ "$CURRENT_MTIME" -gt "$LAST_MTIME" ]; then
        LAST_MTIME=$CURRENT_MTIME
        LAST_GROWTH_TIME=$NOW
    fi

    STALE_FOR=$(( NOW - LAST_GROWTH_TIME ))
    if [ "$STALE_SECONDS" -gt 0 ] && [ "$STALE_FOR" -ge "$STALE_SECONDS" ]; then
        log "Log stale for ${STALE_FOR}s (threshold ${STALE_SECONDS}s). Killing build."

        TS=$(date '+%Y-%m-%d %H:%M:%S')
        printf '\n[%s] === KILLED BY WATCHDOG (log stale for %ss, threshold %ss) ===\n' \
            "$TS" "$STALE_FOR" "$STALE_SECONDS" >> "$LOG_PATH" 2>/dev/null || true

        # Kill the whole GROUP when the orchestrator leads its own process
        # group (launch_build.sh starts it under setsid). A parent-walk
        # misses grandchildren — crew agents' uv -> serena -> pyright
        # chains — and SIGKILL reparents them to init (measured 2026-08-13:
        # a killed build left a 21-hour five-server serena quintet). Group
        # kill ONLY when the orchestrator is the leader: a hand-launched
        # build shares the launching shell's group, and killing that group
        # would take the shell (and possibly this watchdog) with it.
        ORCH_PGID=$(ps -o pgid= -p "$ORCH_PID" 2>/dev/null | tr -d ' ')
        WD_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
        if [ -n "$ORCH_PGID" ] && [ "$ORCH_PGID" = "$ORCH_PID" ] && \
           [ "$ORCH_PGID" != "$WD_PGID" ]; then
            log "Killing process group $ORCH_PGID (orchestrator is leader)"
            kill -9 -- "-$ORCH_PGID" 2>/dev/null || true
        else
            log "Orchestrator $ORCH_PID is not a group leader (pgid" \
                "${ORCH_PGID:-unknown}); parent-walk kill — grandchildren" \
                "may survive (hand launch without setsid)."
            CHILDREN=$(pgrep -P "$ORCH_PID" 2>/dev/null || true)
            if [ -n "$CHILDREN" ]; then
                log "Killing children: $CHILDREN"
                kill -9 $CHILDREN 2>/dev/null || true
            fi
            log "Killing orchestrator: $ORCH_PID"
            kill -9 "$ORCH_PID" 2>/dev/null || true
        fi

        if [ "${AGENT_PROVIDER:-claude}" = "codex" ]; then
            SERENA_PORT="${CODEX_SERENA_PORT:-8324}"
        elif [ "${AGENT_PROVIDER:-claude}" = "opencode" ]; then
            SERENA_PORT="${OPENCODE_SERENA_PORT:-8325}"
        else
            SERENA_PORT="${SDK_SERENA_PORT:-8322}"
        fi
        SERENA_PID=$(lsof -tiTCP:"$SERENA_PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)
        if [ -n "$SERENA_PID" ]; then
            log "Killing orphaned Serena on port $SERENA_PORT: PID $SERENA_PID"
            kill -9 "$SERENA_PID" 2>/dev/null || true
        fi

        log "Watchdog kill complete. Exit code 2."
        exit 2
    fi
done
