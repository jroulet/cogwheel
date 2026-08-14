#!/usr/bin/env bash
# .claude/sdk/run_full_suite.sh — the post-build full-suite gate, in one
# hook-allowlisted command.
#
# Usage: .claude/sdk/run_full_suite.sh [log_path]
#
# Exists because the gate recipe lived only as prose in launch_build.sh's
# post-build banner, so every driver run retyped it and re-derived the same
# details: --collect-only first, -n 8 --dist loadfile (never serial), timing
# guards deselected then run in one serial pass, slow tiers pinned OFF.
#
# SELF-EMITS progress beats on stdout so the run needs no instrumentation
# (CLAUDE.md "Testing / driver discipline"): a beat on CHANGE only, plus one
# on entering a stall and one at terminal.
#
# Exit codes: 0 = all green; 1 = failures; 2 = collection mismatch.
set -u

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 2
LOG="${1:-/tmp/full_suite_$(date +%Y%m%d_%H%M%S).log}"

# Slow tiers NEVER run here — this is the FAST gate. The slow sweeps are a
# separate driver step (.claude/sdk/post_build_sweeps.sh).
export COGWHEEL_BRUTE_ACCURACY="" COGWHEEL_TRAIN_TIER="" \
       COGWHEEL_STRICT_TIMING="" COGWHEEL_RUN_TIMING_SMOKE=""

# Under a pty (background task wrappers) pytest colorizes, and ANSI codes
# in front of "N tests collected" broke the collection grep below — the
# gate then reported "collection errored" on a healthy 2267-test collect
# (2026-08-14). Force plain output everywhere.
export PY_COLORS=0

if [[ -f "$REPO_ROOT/.env" ]] && [[ -z "${SDK_CONDA_ENV:-}" ]]; then
  SDK_CONDA_ENV="$(grep -E '^SDK_CONDA_ENV=' "$REPO_ROOT/.env" |
                   cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
ENV_NAME="${SDK_CONDA_ENV:?SDK_CONDA_ENV is not set — copy .env.example to .env and set it}"
PYBIN="$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "ERROR: $PYBIN not found/executable; check SDK_CONDA_ENV" >&2
  exit 2
fi

# xdist workers SHARE one cache dir; independent pytest processes get one each.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_full_suite_$$}"
mkdir -p "$NUMBA_CACHE_DIR"

echo "[gate] log=$LOG env=$ENV_NAME"

# 1. Collection must succeed and be counted BEFORE the run: a silent
#    collect-error (the missing IMRPhenomXODE symlink in a fresh worktree)
#    otherwise reads as a smaller, greener suite.
COLLECTED=$("$PYBIN" -m pytest cogwheel/tests/ -q --collect-only 2>&1 |
            grep -oE '^[0-9]+ tests collected' | grep -oE '^[0-9]+')
if [[ -z "$COLLECTED" ]]; then
  echo "[gate] FAIL: collection errored — not running the suite" >&2
  "$PYBIN" -m pytest cogwheel/tests/ -q --collect-only 2>&1 | tail -20 >&2
  exit 2
fi
echo "[gate] collected $COLLECTED tests"
if [[ -n "${EXPECT_COLLECTED:-}" ]] && [[ "$COLLECTED" != "$EXPECT_COLLECTED" ]]
then
  echo "[gate] FAIL: expected $EXPECT_COLLECTED tests, collected $COLLECTED" >&2
  exit 2
fi

# 2. Parallel pass. -v so progress is countable; loadscope so same-scope tests
#    share a worker (module-scope fixtures and the numba cache).
#    -n 4 (not 8): 8 workers hit OOM during numba JIT compilation, crashing
#    xdist with INTERNALERROR/MemoryError (diagnosed 2026-08-03 in tree_gate.log).
#
#    --timeout: an unbounded test must FAIL LOUDLY AND NAME ITSELF, not pin
#    workers until something else gives up. On 2026-08-05 four tests entered
#    f_schwinger's mpmath band and never returned; the gate sat at 99% for six
#    hours and the next run's tree gate hit its own 3600s ceiling and STRANDED
#    a build. Neither run named a single test -- both needed a py-spy autopsy.
#    method=signal (not thread): SIGALRM interrupts the offending test only and
#    the run continues; thread kills the whole worker process, losing the rest
#    of its scope. The mpmath path is pure Python, so SIGALRM lands.
TEST_TIMEOUT="${GATE_TEST_TIMEOUT:-600}"
"$PYBIN" -m pytest cogwheel/tests/ -v -n 4 --dist loadscope \
  --timeout="$TEST_TIMEOUT" --timeout-method=signal \
  -k "not Timing and not timing" > "$LOG" 2>&1 &
PYTEST_PID=$!

# Poll often, ANNOUNCE a stall rarely. The slow tail (test_posterior, the
# waveform suites) legitimately goes minutes between result lines, so a
# one-poll stall threshold cried wolf twice on a run that was healthy at 2448%
# worker CPU and finished green. A stall notice that fires on healthy runs
# trains the reader to ignore the channel, which is the whole value of it.
LAST=0; QUIET=0; STALL_ANNOUNCED=0
STALL_AFTER="${GATE_STALL_AFTER:-300}"
while kill -0 "$PYTEST_PID" 2>/dev/null; do
  sleep 30
  DONE=$(grep -acE '(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)' "$LOG" 2>/dev/null)
  if [[ "$DONE" -gt "$LAST" ]]; then
    LAST=$DONE; QUIET=0; STALL_ANNOUNCED=0
    echo "[beat] $DONE/$COLLECTED ($(( DONE * 100 / COLLECTED ))%)"
  else
    QUIET=$(( QUIET + 30 ))
    if [[ "$QUIET" -ge "$STALL_AFTER" ]] && [[ "$STALL_ANNOUNCED" -eq 0 ]]; then
      STALL_ANNOUNCED=1
      echo "[beat] STALL at $DONE/$COLLECTED — no progress for ${QUIET}s"
    fi
  fi
done
wait "$PYTEST_PID"; PAR_RC=$?
echo "[gate] parallel pass rc=$PAR_RC"
grep -aE '^(FAILED|ERROR)' "$LOG" | head -30

# 3. The deselected timing guards, in ONE serial pass.
echo "[beat] timing guards (serial)"
"$PYBIN" -m pytest cogwheel/tests/ -q -k "Timing or timing" >> "$LOG" 2>&1
TIM_RC=$?
echo "[gate] timing pass rc=$TIM_RC"

echo "[gate] === SUMMARY ==="
grep -aE '^[0-9]+ (passed|failed)|passed|failed' "$LOG" | tail -4
if [[ "$PAR_RC" -eq 0 ]] && [[ "$TIM_RC" -eq 0 ]]; then
  echo "[gate] ALL GREEN ($COLLECTED collected)"; exit 0
fi
echo "[gate] GATE FAILURE (parallel rc=$PAR_RC timing rc=$TIM_RC)"; exit 1
