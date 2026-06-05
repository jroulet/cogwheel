#!/usr/bin/env bash
# .claude/sdk/watchdog.sh — kill a wedged SDK build if its log goes stale.
#
# Usage:
#   watchdog.sh <log_path> [stale_seconds]
#
# Polls <log_path> every 30s. If the file's mtime hasn't advanced in
# <stale_seconds> (default 600 = 10 min), kills the orchestrator subtree.
# Self-exits cleanly when the orchestrator exits naturally.
#
# Pass stale_seconds=0 to disable the staleness check while still
# tracking the orchestrator process.
#
# Exit codes:
#   0 = orchestrator exited normally
#   1 = setup error
#   2 = watchdog killed the build

set -u

LOG_PATH="${1:?usage: watchdog.sh <log_path> [stale_seconds]}"
STALE_SECONDS="${2:-600}"
POLL_INTERVAL=30
STARTUP_GRACE=60

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

# Find the orchestrator PID
ORCH_PID=$(pgrep -nf '\.claude/sdk/cli\.py build' 2>/dev/null || true)
if [ -z "$ORCH_PID" ]; then
    log "ERROR: Could not find orchestrator process. Exiting."
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

        CHILDREN=$(pgrep -P "$ORCH_PID" 2>/dev/null || true)
        if [ -n "$CHILDREN" ]; then
            log "Killing children: $CHILDREN"
            kill -9 $CHILDREN 2>/dev/null || true
        fi

        log "Killing orchestrator: $ORCH_PID"
        kill -9 "$ORCH_PID" 2>/dev/null || true

        SERENA_PID=$(lsof -tiTCP:8322 -sTCP:LISTEN 2>/dev/null | head -1 || true)
        if [ -n "$SERENA_PID" ]; then
            log "Killing orphaned Serena on port 8322: PID $SERENA_PID"
            kill -9 "$SERENA_PID" 2>/dev/null || true
        fi

        log "Watchdog kill complete. Exit code 2."
        exit 2
    fi
done
