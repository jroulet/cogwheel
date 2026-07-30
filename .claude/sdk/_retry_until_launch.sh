#!/usr/bin/env bash
# Driver retry loop: relaunch an SDK build every GAP seconds until one
# survives the subagent fan-out that is currently failing post-outage
# (single API calls work; concurrent subagent spawns die at ~25-30s with
# "message reader exit 1"). Stops as soon as a build reaches its plan gate
# (writes plan.json) OR clearly progresses past planning — at which point
# the DRIVER must take over to approve the gate. Also stops on repeated
# clean successes or the attempt cap.
#
# Emits one stdout line per state change (Monitor-friendly). Not part of the
# pipeline; safe to delete.
set -u
SLUG="arc_guard_fix"
BRIEF="/home/tejaswi/Work/cogwheel-claude-dev/.claude/handoff/brief_arc_guard_fix.md"
GAP="${GAP:-900}"          # 15 min between tries
MAX="${MAX:-16}"           # ~4h ceiling
POLL_SECS=180             # how long to watch each attempt for surv/death
cd /home/tejaswi/Work/cogwheel-claude-dev || exit 1

attempt=0
while [ "$attempt" -lt "$MAX" ]; do
    attempt=$((attempt + 1))
    OUT="$(.claude/sdk/launch_build.sh "$SLUG" "$BRIEF" 2>&1)"
    LOG="$(printf '%s\n' "$OUT" | sed -n 's/^launched: \([^ ]*\).*/\1/p' | head -1)"
    if [ -z "$LOG" ]; then
        echo "attempt $attempt: LAUNCH produced no log path — $(printf '%s' "$OUT" | head -1)"
        sleep "$GAP"; continue
    fi
    # The build is detached; its log file appears a beat after launch_build.sh
    # returns. Wait for it rather than racing (the bug that launched an
    # unmonitored build on the first go).
    fwait=0
    while [ ! -f "$LOG" ] && [ "$fwait" -lt 20 ]; do sleep 1; fwait=$((fwait + 1)); done
    if [ ! -f "$LOG" ]; then
        echo "attempt $attempt: log $LOG never appeared after 20s — treating as failed"
        sleep "$GAP"; continue
    fi
    echo "attempt $attempt launched: $LOG"

    # Watch this attempt. SURVIVAL is keyed on the plan-gate ARTIFACT existing,
    # not on log text -- launch_build.sh echoes a suggested Monitor command
    # into the log header that literally contains the words "Plan written",
    # "plan_ready", etc., so grepping the log for those matched the header and
    # false-reported success. The approval file is unambiguous.
    # DEATH is keyed on real failure lines that do NOT appear in that header
    # echo ("Fatal error in message reader" / a line beginning "Build failed:").
    APPROVAL_DIR="/tmp/${SLUG}_approval"
    waited=0
    verdict=""
    while [ "$waited" -lt "$POLL_SECS" ]; do
        sleep 5; waited=$((waited + 5))
        if [ -f "$APPROVAL_DIR/plan.json" ] || [ -f "$APPROVAL_DIR/plan_ready" ]; then
            verdict="SURVIVED"; break
        fi
        if grep -qE "Fatal error in message reader|^Build failed:" "$LOG" 2>/dev/null; then
            verdict="DIED"; break
        fi
    done

    if [ "$verdict" = "SURVIVED" ]; then
        echo "attempt $attempt SURVIVED fan-out -> plan gate reached. DRIVER: approve at $LOG"
        echo "RETRY_LOOP_DONE $LOG"
        exit 0
    fi

    if [ "$verdict" = "DIED" ]; then
        echo "attempt $attempt died at fan-out (post-outage concurrency). waiting ${GAP}s before retry."
    else
        # Neither marker within POLL_SECS. If the process is still alive it is
        # most likely progressing (survived the fast fan-out death); hand off.
        if pgrep -fc "[s]dk/build.py" >/dev/null 2>&1; then
            echo "attempt $attempt still alive after ${POLL_SECS}s with no death marker -> likely progressing. DRIVER: check $LOG"
            echo "RETRY_LOOP_DONE $LOG"
            exit 0
        fi
        echo "attempt $attempt: inconclusive and no build process; treating as died. waiting ${GAP}s."
    fi
    sleep "$GAP"
done
echo "RETRY_LOOP_EXHAUSTED after $MAX attempts — API concurrency still not recovered; escalate to user."
exit 1
