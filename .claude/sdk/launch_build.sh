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
# blocked). Override the conda env via SDK_CONDA_ENV or a repo-root .env
# (copy .env.example to .env); shell env wins, then .env, then cogwheel_310.
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

# Conda env routing via the durable .env idiom (mirrors gw_detection_ias):
# shell SDK_CONDA_ENV wins, then .env at the repo root, then the default —
# same precedence as python-dotenv's load_dotenv(override=False). Copy
# .env.example to .env to set it per machine.
if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "${SDK_CONDA_ENV:-}" ]]; then
  SDK_CONDA_ENV="$(grep -E '^SDK_CONDA_ENV=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
ENV_NAME="${SDK_CONDA_ENV:?SDK_CONDA_ENV is not set — copy .env.example to .env and set it}"

# Per-repo Serena SSE port, same .env precedence (shell > .env > 8322).
# Sibling pipelines (gw) hardcode 8322 and their watchdogs kill any 8322
# listener — set SDK_SERENA_PORT in .env to keep the repos disjoint.
# Exported so build.py (SerenaManager) and watchdog.sh both see it.
if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "${SDK_SERENA_PORT:-}" ]]; then
  SDK_SERENA_PORT="$(grep -E '^SDK_SERENA_PORT=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
export SDK_SERENA_PORT="${SDK_SERENA_PORT:-8322}"

# Orchestrator reliability knobs, same .env precedence (shell > .env >
# orchestrator default). INTER_MESSAGE_TIMEOUT: the 300s default
# misclassifies long single-turn deliberation as a stall and killed
# Build 6 coders twice (2026-07-18). SKIP_TIDIER: a tidier error_max_turns
# escapes the graceful catch via the anyio cancel-scope bug and kills the
# whole DAG (2/2 reproduced, Build 6 attempts 5-6).
for KNOB in SDK_INTER_MESSAGE_TIMEOUT_SECONDS SDK_SKIP_TIDIER; do
  if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "$(eval echo "\${$KNOB:-}")" ]]; then
    VAL="$(grep -E "^$KNOB=" "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
    [[ -n "$VAL" ]] && export "$KNOB=$VAL"
  fi
done

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

# Resolve the env's python ABSOLUTELY: `conda run ... python` trusts PATH,
# and under some shells (serena MCP) a uv shim shadows the env python,
# yielding ModuleNotFoundError for claude_agent_sdk (build 2d, 2026-07-17).
PYBIN="$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "ERROR: $PYBIN not found/executable; check SDK_CONDA_ENV" >&2
  exit 1
fi

"$PYBIN" "$REPO_ROOT/.claude/sdk/build.py" build "${APPROVE_ARGS[@]}" \
  --log "$LOG" "@$PROMPT" > /dev/null 2>&1 &
BUILD_PID=$!
# Hand the watchdog the PID we already know. It used to rediscover the
# orchestrator by pattern, and the pattern named cli.py while the entrypoint
# is build.py (a shim that calls cli.main() in-process, so argv never says
# "cli.py") — every launcher-started build from 2026-07-27 ran unguarded
# (F055). A passed PID also removes `pgrep -n`'s newest-match race, which
# pointed both watchdogs at the same build whenever two ran at once.
"$REPO_ROOT/.claude/sdk/watchdog.sh" "$LOG" "$STALE" "$BUILD_PID" \
  > /dev/null 2>&1 &
WATCHDOG_PID=$!
disown -a

# Verify the watchdog ATTACHED before claiming protection. The silent version
# of this failure is what let F055 run for three days: the launcher printed
# "(watchdog 1200s)" while the watchdog had already exited 1 into a sidecar
# log nobody reads. Loop exits early both ways — on attach, or on its death.
WATCHDOG_LOG="${LOG%.log}.watchdog.log"
for _ in $(seq 1 70); do
  grep -q "Watching orchestrator PID" "$WATCHDOG_LOG" 2>/dev/null && break
  kill -0 "$WATCHDOG_PID" 2>/dev/null || break
  sleep 1
done
if grep -q "Watching orchestrator PID $BUILD_PID" "$WATCHDOG_LOG" 2>/dev/null
then
  echo "launched: $LOG (watchdog ${STALE}s, guarding PID $BUILD_PID)"
else
  echo "launched: $LOG"
  echo "WARNING: WATCHDOG DID NOT ATTACH — this build has no kill" \
       "protection. See $WATCHDOG_LOG" >&2
fi
# The post-build SEQUENCE, not just the sweeps.  Librarian and Dreamer run
# IN the DAG; the Tidier is the one crew role with no automated home, since
# its in-DAG run was made opt-in on 2026-07-18 after an error_max_turns
# finalization tore down the whole DAG group.  A single unmechanised step in
# an otherwise automated pipeline is exactly what goes unnoticed -- it lapsed
# for nine days -- so print it where the driver actually looks.
cat <<'POSTBUILD'
POST-BUILD SEQUENCE (driver steps -- the DAG does NOT run these):
  1. full-suite tally   python -m pytest cogwheel/tests/ -q -n 8 --dist loadfile \
                          -k "not Timing and not timing"   (then timing serially)
  2. commit the build's work
  3. slow sweeps        .claude/sdk/post_build_sweeps.sh   # slow tiers NEVER in-build
  4. /tidy              then: python scripts/update_agent_state.py tidy
                        (in-DAG tidier is opt-in, SDK_RUN_TIDIER=1, default OFF --
                         it tore down the DAG group on 2026-07-18, so style is a
                         post-commit advisory role and this step is the ONLY one
                         that consumes .claude/tidy_advisory.json)

IN-DAG (the build runs these itself -- do NOT re-run on a clean build):
  Librarian, Dreamer.
  EXCEPTION: if the build STRANDED without committing, its Librarian/Dreamer
  work is not on disk or not committed. Then the driver runs them by hand:
    /doc-sync --post-commit <sha>   # also clears the pre-commit backlog guard
    /dream                          # short-term memories are tail-capped (F021):
                                    # a busy day evicts findings before consolidation
POSTBUILD
echo "arm the Monitor from the log header: tail -20 $LOG once it exists"
if [[ "$AUTO" != "--auto" ]]; then
  echo "PLAN APPROVAL: on the plan-ready log line,"
  echo "  read:    $APPROVAL_DIR/plan.json"
  echo "  approve: touch $APPROVAL_DIR/plan_approved"
  echo "  reject:  write feedback to $APPROVAL_DIR/plan_rejected"
  echo "Respond promptly: the watchdog staleness clock runs during the wait."
fi
