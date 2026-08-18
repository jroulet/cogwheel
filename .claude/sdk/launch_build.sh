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
USAGE="  env: AGENT_PROVIDER=claude|codex|opencode (default: claude)"
SLUG="${1:?$USAGE}"
PROMPT="${2:?$USAGE}"
STALE="${3:-1200}"
AUTO="${4:-}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Provider selection — defaults to claude; set AGENT_PROVIDER in the shell
# or in .env to use codex/opencode. Exports so build.py inherits it.
#
# Precedence is shell > .env > "claude", matching load_dotenv(override=False).
# That precedence is correct but its failure mode was SILENT: an exported
# AGENT_PROVIDER (or OPENCODE_MODEL_PROVIDER) outranks the file, so you could
# edit .env, launch, and get the other backend with nothing in the output
# saying so. Measured 2026-08-12: a stale exported OPENCODE_MODEL_PROVIDER=go
# in one shell kept selecting deepseek models after .env had been set to the
# Claude models. So: resolve, then SAY what won and why.
_prov_shell="${AGENT_PROVIDER:-}"
_prov_env=""
if [[ -f "$REPO_ROOT/.env" ]]; then
  _prov_env="$(grep -E '^AGENT_PROVIDER=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
if [[ -z "$_prov_shell" && -n "$_prov_env" ]]; then
  AGENT_PROVIDER="$_prov_env"
fi
export AGENT_PROVIDER="${AGENT_PROVIDER:-claude}"
if [[ -n "$_prov_shell" ]]; then
  _prov_src="shell env"
  if [[ -n "$_prov_env" && "$_prov_env" != "$_prov_shell" ]]; then
    echo "WARNING: AGENT_PROVIDER='$_prov_shell' from the SHELL overrides" \
         "'$_prov_env' in .env — the file is being ignored. unset it to use" \
         ".env." >&2
  fi
elif [[ -n "$_prov_env" ]]; then
  _prov_src=".env"
else
  _prov_src="default"
fi
echo "provider: $AGENT_PROVIDER (from $_prov_src)"

# Same silent-override hazard for the OpenCode model tier, which decides
# whether roles run on Claude or DeepSeek models.
if [[ "$AGENT_PROVIDER" == "opencode" && -n "${OPENCODE_MODEL_PROVIDER:-}" \
      && -f "$REPO_ROOT/.env" ]]; then
  _omp_env="$(grep -E '^OPENCODE_MODEL_PROVIDER=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
  if [[ "$_omp_env" != "${OPENCODE_MODEL_PROVIDER}" ]]; then
    echo "WARNING: OPENCODE_MODEL_PROVIDER='${OPENCODE_MODEL_PROVIDER}' from" \
         "the SHELL overrides '${_omp_env}' in .env — role models come from" \
         "the shell value." >&2
  fi
fi

# Ensure provider CLIs are on PATH (opencode installs to ~/.opencode/bin,
# codex to ~/.local/bin or similar — non-interactive shells may miss them).
for _bindir in "$HOME/.opencode/bin" "$HOME/.local/bin"; do
  [[ -d "$_bindir" ]] && [[ ":$PATH:" != *":$_bindir:"* ]] && export PATH="$_bindir:$PATH"
done

# OpenCode session and auth from .env (same precedence: shell > .env > default).
# OPENCODE_SESSION_ID is REQUIRED for the callback sidecar to know where to
# inject the terminal event. Without it, builds complete silently.
if [[ "$AGENT_PROVIDER" == "opencode" && -f "$REPO_ROOT/.env" ]]; then
  for _VAR in OPENCODE_SESSION_ID OPENCODE_SERVER_USERNAME OPENCODE_SERVER_PASSWORD OPENCODE_SERVE_PORT OPENCODE_MODEL_PROVIDER; do
    if [[ -z "$(eval echo "\${$_VAR:-}")" ]]; then
      _VAL="$(grep -E "^$_VAR=" "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
      [[ -n "$_VAL" ]] && export "$_VAR=$_VAL"
    fi
  done
fi

# Auto-start opencode serve if needed for callbacks. The serve API is how the
# resume driver injects messages that trigger agent turns. Without it, builds
# complete but can't notify the interactive driver.
if [[ "$AGENT_PROVIDER" == "opencode" ]]; then
  SERVE_PORT="${OPENCODE_SERVE_PORT:-4096}"
  if ! curl -sf "http://localhost:$SERVE_PORT/global/health" -u "${OPENCODE_SERVER_USERNAME:-opencode}:${OPENCODE_SERVER_PASSWORD:-}" >/dev/null 2>&1; then
    if [[ -z "${OPENCODE_SERVER_PASSWORD:-}" ]]; then
      echo "WARNING: OPENCODE_SERVER_PASSWORD not set — serve API callbacks won't work" >&2
    else
      echo "Starting opencode serve on port $SERVE_PORT..."
      # Must run from REPO_ROOT so it loads .opencode/opencode.json
      # (MCP/Serena config, permissions, instructions).
      # Use setsid to create a new process group — the calling shell's
      # tool timeout kills the foreground process GROUP, and without setsid
      # the serve process is in that group and dies with it.
      setsid bash -c "cd '$REPO_ROOT' && OPENCODE_SERVER_PASSWORD='${OPENCODE_SERVER_PASSWORD}' \
        opencode serve --port $SERVE_PORT --hostname 127.0.0.1" \
        > /tmp/opencode_serve.log 2>&1 &
      _SERVE_PID=$!
      disown $_SERVE_PID
      # Wait briefly for it to come up
      for _ in $(seq 1 10); do
        curl -sf "http://localhost:$SERVE_PORT/global/health" \
          -u "${OPENCODE_SERVER_USERNAME:-opencode}:${OPENCODE_SERVER_PASSWORD}" \
          >/dev/null 2>&1 && break
        sleep 1
      done
      if curl -sf "http://localhost:$SERVE_PORT/global/health" \
          -u "${OPENCODE_SERVER_USERNAME:-opencode}:${OPENCODE_SERVER_PASSWORD}" \
          >/dev/null 2>&1; then
        echo "opencode serve ready (PID $_SERVE_PID, port $SERVE_PORT)"
      else
        echo "WARNING: opencode serve failed to start — callbacks may not work" >&2
      fi
    fi
  fi
fi

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

# Provider-specific Serena ports (distinct so simultaneous builds don't collide).
if [[ "$AGENT_PROVIDER" == "codex" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "${CODEX_SERENA_PORT:-}" ]]; then
    CODEX_SERENA_PORT="$(grep -E '^CODEX_SERENA_PORT=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
  fi
  export CODEX_SERENA_PORT="${CODEX_SERENA_PORT:-8324}"
elif [[ "$AGENT_PROVIDER" == "opencode" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "${OPENCODE_SERENA_PORT:-}" ]]; then
    OPENCODE_SERENA_PORT="$(grep -E '^OPENCODE_SERENA_PORT=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")"
  fi
  export OPENCODE_SERENA_PORT="${OPENCODE_SERENA_PORT:-8325}"
fi

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

# Zero-cost planning recovery: SDK_RESUME_PLAN=<path to plan.json> loads a
# previously approved/crash-recovery plan and skips the Architect (cli.py
# --resume-plan; verification + approval gates still run).
if [[ -n "${SDK_RESUME_PLAN:-}" ]]; then
  APPROVE_ARGS+=(--resume-plan "$SDK_RESUME_PLAN")
fi

# Resolve the env's python ABSOLUTELY: `conda run ... python` trusts PATH,
# and under some shells (serena MCP) a uv shim shadows the env python,
# yielding ModuleNotFoundError for claude_agent_sdk (build 2d, 2026-07-17).
PYBIN="$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "ERROR: $PYBIN not found/executable; check SDK_CONDA_ENV" >&2
  exit 1
fi

# Overlap guard (2026-08-14): a failed build can legitimately keep running
# its Phase-3 memory consolidation for minutes after its terminal event
# fires, holding the project's SSE serena port. Launching a second build
# then SHARES the old build's serena, and the old build's teardown kills
# it under the new one. Detect and warn loudly (not fatal: the new build
# falls back to built-in tools and its SerenaManager can restart, but the
# operator should know the race exists).
_LIVE_ORCH="$(pgrep -f "${REPO_ROOT}/\.claude/sdk/build\.py build" 2>/dev/null | head -3 | tr '\n' ' ')"
if [[ -n "${_LIVE_ORCH// /}" ]]; then
  echo "WARNING: a same-repo orchestrator is STILL RUNNING (PID(s) ${_LIVE_ORCH}) —" \
       "likely post-failure Phase 3. The new build will share (and may lose)" \
       "its serena. Prefer waiting for it to exit." >&2
fi

# Daemon hygiene: reap THIS project's stale serena/pyright chains before
# spawning a new crew — every non-graceful client death leaves a stdio pair
# re-indexing the repo forever (16 serena + 16 pyright had accumulated by
# 2026-08-14 and pinned swap). Project-discriminated: another project's
# serena (e.g. a gw build's) is never touched. Then count what is still
# live — the daemon-count analogue of the chart cell-count guard: count
# cheaply at the entry point, never discover as mystery latency.
"$PYBIN" "$REPO_ROOT/.claude/sdk/reap_stale_serena.py" --apply || true
_N_SERENA=$("$PYBIN" "$REPO_ROOT/.claude/sdk/reap_stale_serena.py" --count-live 2>/dev/null || echo "?")
if [[ "$_N_SERENA" =~ ^[0-9]+$ ]] && (( _N_SERENA > 3 )); then
  echo "WARNING: $_N_SERENA live serena servers already serve THIS project" \
       "(threshold 3) and the build will add its own. If unexpected, inspect:" \
       "  $PYBIN $REPO_ROOT/.claude/sdk/reap_stale_serena.py   (dry-run)" >&2
fi

# Sync .opencode/agents/*.md frontmatter models from the env-selected role
# maps so interactive subagents match the build provider.  Single source of
# truth: runtime_opencode.py; no manual frontmatter edits ever needed.
if [[ "$AGENT_PROVIDER" == "opencode" ]]; then
  # Do NOT swallow stderr: the sync script's only output is the line naming
  # the resolved model provider AND its source, added in 4c16ca4 precisely to
  # expose a silent ai-commons/go mis-selection. Discarding it here left that
  # fix working only for a bare manual run and broken on the automated path
  # that actually runs before every build (found 2026-08-12).
  "$PYBIN" "$REPO_ROOT/scripts/sync_opencode_agents.py" || true
fi

# Same for Codex: sync .codex/agents/*.toml model fields from the role maps
# in runtime_codex.py so interactive Codex subagents match SDK builds.
if [[ "$AGENT_PROVIDER" == "codex" ]]; then
  # stderr kept for the same reason as the OpenCode sync above.
  "$PYBIN" "$REPO_ROOT/scripts/sync_codex_agents.py" || true
fi
# setsid: the orchestrator gets its OWN process group so the watchdog can
# kill the GROUP on a stall — a parent-walk kill misses grandchildren
# (crew agents' uv -> serena -> pyright chains) and SIGKILL reparents them
# to init; a killed build left a 21-hour five-server serena quintet
# (2026-08-13). Non-interactive bash runs & jobs in the script's own
# group, so setsid execs in place and $! below IS the orchestrator PID.
setsid "$PYBIN" "$REPO_ROOT/.claude/sdk/build.py" build --provider "$AGENT_PROVIDER" \
  "${APPROVE_ARGS[@]}" --log "$LOG" "@$PROMPT" > /dev/null 2>&1 &
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

# Terminal callback sidecar: waits for the build to exit and fires the
# provider-appropriate resume driver so the interactive agent is notified.
# Must be spawned BEFORE disown so it can wait on BUILD_PID (its sibling).
_RESUME_HELPER=""
if [[ "$AGENT_PROVIDER" == "codex" && -n "${CODEX_THREAD_ID:-}" && "${CODEX_EVENT_RESUME:-1}" != "0" ]]; then
  _RESUME_HELPER="$REPO_ROOT/.codex/resume_driver.sh"
elif [[ "$AGENT_PROVIDER" == "opencode" && -n "${OPENCODE_SESSION_ID:-}" && "${OPENCODE_EVENT_RESUME:-1}" != "0" ]]; then
  _RESUME_HELPER="$REPO_ROOT/.opencode/resume_driver.sh"
fi
if [[ -n "$_RESUME_HELPER" && -f "$_RESUME_HELPER" ]]; then
  _RESUME_EVENT_ID="build-$(date +%s%N)-$$-${RANDOM}"
  # Can't use `wait` — after disown the build PID is no longer a child.
  # Poll /proc/<pid> instead (lightweight, no busy-wait — 2s interval).
  (
    while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 2; done
    # Recover exit status from the log's terminal marker if possible.
    if grep -qaE "^(\[[^]]*\])?\s*(===\s*)?(BUILD REPORT|Build failed|Build cancelled|KILLED BY WATCHDOG)" "$LOG" 2>/dev/null; then
      _STATUS=0
      grep -qa "Build failed\|KILLED BY WATCHDOG\|Build cancelled" "$LOG" 2>/dev/null && _STATUS=1
    else
      _STATUS=1
    fi
    "$_RESUME_HELPER" build_terminal \
      "exit_status=$_STATUS log=$LOG" "$_RESUME_EVENT_ID"
  ) </dev/null >/dev/null 2>&1 &
fi

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
if [[ -n "$_RESUME_HELPER" ]]; then
  echo "terminal callback armed ($AGENT_PROVIDER)"
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
                         NOTE: a SCOPED post-build tidier now runs by default
                         after commit (mechanical + judgment on the build's changed
                         .py files). Skip it with SDK_SKIP_TIDIER=1.

IN-DAG (the build runs these itself -- do NOT re-run on a clean build):
  Librarian, Dreamer.
  EXCEPTION: if the build STRANDED without committing, its Librarian/Dreamer
  work is not on disk or not committed. Then the driver runs them by hand:
    /doc-sync --post-commit <sha>   # also clears the pre-commit backlog guard
    /dream                          # short-term memories are tail-capped (F021):
                                    # a busy day evicts findings before consolidation
POSTBUILD
echo "ARM THIS MONITOR (persistent=true) -- do NOT hand-roll one:
  .claude/sdk/build_monitor.sh $LOG 120 $BUILD_PID"
if [[ "$AUTO" != "--auto" ]]; then
  echo "PLAN APPROVAL: on the plan-ready log line,"
  echo "  read:    $APPROVAL_DIR/plan.json"
  echo "  approve: touch $APPROVAL_DIR/plan_approved"
  echo "  reject:  write feedback to $APPROVAL_DIR/plan_rejected"
  echo "Respond promptly: the watchdog staleness clock runs during the wait."
fi

