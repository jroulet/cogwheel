#!/usr/bin/env bash
# Resume one OpenCode driver session after a terminal build event or escalation.

set -uo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 EVENT DETAIL [EVENT_ID]" >&2
  exit 64
fi

EVENT="$1"
DETAIL="$2"
EVENT_ID="${3:-${OPENCODE_BUILD_EVENT_ID:-}}"
SESSION_ID="${OPENCODE_SESSION_ID:-}"

if [[ -z "$SESSION_ID" ]] || [[ "${OPENCODE_EVENT_RESUME:-1}" == "0" ]]; then
  exit 0
fi
if [[ ! "$SESSION_ID" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "Invalid OPENCODE_SESSION_ID; refusing callback." >&2
  exit 65
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
STATE_DIR="${OPENCODE_RESUME_STATE_DIR:-/tmp/cogwheel-opencode-resume}"
LOG_FILE="${OPENCODE_RESUME_LOG:-$STATE_DIR/resume.log}"
mkdir -p "$STATE_DIR"

# New callers provide a stable occurrence identity. The fallback protects an
# older already-running SDK process that still supplies only event + detail:
# later same-path events must wake rather than being mistaken for duplicates.
if [[ -z "$EVENT_ID" ]]; then
  EVENT_ID="legacy-$(date +%s%N)-$$-${RANDOM}"
fi
SESSION_KEY="$(printf '%s' "$SESSION_ID" | sha256sum | awk '{print $1}')"
EVENT_KEY="$(printf '%s\0%s\0%s' "$EVENT" "$DETAIL" "$EVENT_ID" \
  | sha256sum | awk '{print $1}')"
LOCK_FILE="$STATE_DIR/session-$SESSION_KEY.lock"
DONE_FILE="$STATE_DIR/session-$SESSION_KEY-event-$EVENT_KEY.done"

exec 9>"$LOCK_FILE"
# Queue distinct events. A non-blocking flock silently discarded the second
# escalation in Build 8h-b3 when the first callback still held the lock.
flock 9
[[ -f "$DONE_FILE" ]] && exit 0

PROMPT="A detached Cogwheel build emitted event '$EVENT' (id '$EVENT_ID'): $DETAIL

This is an event callback, not a periodic monitor. Inspect authoritative state
once and continue the existing task. For an escalation, resolve or present it;
for terminal status, evaluate the result and take the next justified action.
Do not create a persistent goal or poll unchanged state."

IFS=',' read -r -a RETRY_DELAYS \
  <<< "${OPENCODE_RESUME_RETRY_DELAYS:-0,30,120}"
for DELAY in "${RETRY_DELAYS[@]}"; do
  if [[ ! "$DELAY" =~ ^[0-9]+$ ]]; then
    echo "Invalid OPENCODE_RESUME_RETRY_DELAYS entry: $DELAY" >&2
    exit 66
  fi
  (( DELAY > 0 )) && sleep "$DELAY"
  {
    printf '\n[%s] event=%s event_id=%s session=%s\n' \
      "$(date --iso-8601=seconds)" "$EVENT" "$EVENT_ID" "$SESSION_ID"
    cd "$REPO_ROOT"
    opencode run --session "$SESSION_ID" --continue --auto -- "$PROMPT"
  } >>"$LOG_FILE" 2>&1
  STATUS=$?
  if (( STATUS == 0 )); then
    touch "$DONE_FILE"
    exit 0
  fi
done

exit "$STATUS"
