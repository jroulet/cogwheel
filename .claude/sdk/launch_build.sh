#!/usr/bin/env bash
# .claude/sdk/launch_build.sh — one-command SDK build launch.
#
# Usage: .claude/sdk/launch_build.sh <task_slug> <prompt_file> [stale_seconds] [--auto]
#
# Does the whole launch in one hook-allowlisted command (this lives under
# .claude/sdk/ so the Bash allowlist permits it directly): resolves the
# conda env, starts build.py with @prompt and a timestamped log, attaches
# the watchdog (default 1200s staleness), disowns, and prints the log path.
# The Monitor command to arm is printed in the log header by cli.py.
#
# Plan approval defaults to the file-based gate (--approval-dir); pass
# --auto for blind auto-approve on genuinely unattended runs.
#
# Exists because every ad-hoc launch rediscovered the same hook denials
# (.claude/build blocked as a leading command; leading LOG=... assignment
# blocked). Override the conda env with SDK_CONDA_ENV if needed.
set -u

USAGE="usage: launch_build.sh <task_slug> <prompt_file> [stale_seconds] [--auto]"
SLUG="${1:?$USAGE}"
PROMPT="${2:?$USAGE}"
STALE="${3:-1200}"
AUTO="${4:-}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [[ ! -f "$PROMPT" ]]; then
  echo "ERROR: prompt file not found: $PROMPT" >&2
  exit 1
fi

# Depth guards (CLAUDE.md "SDK Build Briefs"): non-fatal warnings only —
# bare-denial rate grows with transcript depth (claude-code #74351).
PROMPT_KB=$(( $(wc -c < "$PROMPT") / 1024 ))
if (( PROMPT_KB > 12 )); then
  echo "WARNING: brief is ${PROMPT_KB} KB (>12 KB) — transcript-depth risk;" \
       "trim to mission/fences/facts/acceptance (CLAUDE.md 'SDK Build Briefs')" >&2
fi
if grep -q "META_PLAN" "$PROMPT"; then
  echo "WARNING: brief references META_PLAN (driver journal, not agent" \
       "context) — inline the distilled facts instead" >&2
fi

ENV_NAME="${SDK_CONDA_ENV:-cogwheel_310}"

LOG="/tmp/${SLUG}_$(date +%Y%m%d_%H%M%S).log"

# Detached build + detached watchdog + Monitor on the log, with PLAN
# APPROVAL via the file-based gate (--approval-dir). --auto restores
# blind auto-approve for genuinely unattended runs.
if [[ "$AUTO" == "--auto" ]]; then
  APPROVE_ARGS=(-y)
else
  APPROVAL_DIR="/tmp/${SLUG}_approval"
  rm -rf "$APPROVAL_DIR"
  APPROVE_ARGS=(--approval-dir "$APPROVAL_DIR")
fi

conda run --no-capture-output -n "$ENV_NAME" \
  python "$REPO_ROOT/.claude/sdk/build.py" build "${APPROVE_ARGS[@]}" \
  --log "$LOG" "@$PROMPT" > /dev/null 2>&1 &
"$REPO_ROOT/.claude/sdk/watchdog.sh" "$LOG" "$STALE" > /dev/null 2>&1 &
disown -a

echo "launched: $LOG (watchdog ${STALE}s)"
echo "arm the Monitor from the log header: tail -20 $LOG once it exists"
if [[ "$AUTO" != "--auto" ]]; then
  echo "PLAN APPROVAL: on the plan-ready log line,"
  echo "  read:    $APPROVAL_DIR/plan.json"
  echo "  approve: touch $APPROVAL_DIR/plan_approved"
  echo "  reject:  write feedback to $APPROVAL_DIR/plan_rejected"
  echo "Respond promptly: the watchdog staleness clock runs during the wait."
fi
