#!/usr/bin/env bash
# Post-build slow sweeps — the driver's ONE command after every build
# commit.  Slow tiers NEVER run in-build; this fans them post-build.
# Lessons baked in: width cap + per-process BLAS/OMP thread caps
# (19 uncapped procs x 64 BLAS threads exhausted pthreads on the first
# run), per-process numba cache dirs (shared-cache races segfault),
# skip-green resumability, SWEEP_OUT override to resume a prior dir.
# Usage: .claude/sdk/post_build_sweeps.sh [extra pytest args]
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNPY="$REPO_ROOT/.claude/sdk/run_py.sh"
OUT="${SWEEP_OUT:-/tmp/post_build_sweeps_$(date +%Y%m%d_%H%M%S)}"
MAX_JOBS="${SWEEP_MAX_JOBS:-8}"
mkdir -p "$OUT"

# DISCOVER the tier variables instead of hardcoding them (design ported from
# gw, 2026-07-30). Hardcoding is how F052 happened: the sweep set
# COGWHEEL_BRUTE_ACCURACY and nothing else, so COGWHEEL_TRAIN_TIER -- which
# gates every build's ACCEPTANCE tests -- was run by no job at all. Fixing
# that by hardcoding a second name would leave the same trap for the third.
# Grepping the suite means a new gated file enrolls itself.
#
# PARALLEL_UNSAFE tiers are named explicitly and REPORTED, never silently
# dropped: timing assertions are meaningless under an 8-wide sweep's CPU
# contention, so they belong in a serial pass, not here.
PARALLEL_UNSAFE="COGWHEEL_STRICT_TIMING COGWHEEL_RUN_TIMING_SMOKE"
DISCOVERED=$(grep -rhoE "environ(\.get)?\(?\[?['\"]COGWHEEL_[A-Z_]+['\"]" \
    "$REPO_ROOT"/cogwheel/tests/*.py 2>/dev/null \
    | grep -oE "COGWHEEL_[A-Z_]+" | sort -u)
TIER_ENV=""
SKIPPED_TIERS=""
for v in $DISCOVERED; do
  case " $PARALLEL_UNSAFE " in
    *" $v "*) SKIPPED_TIERS="$SKIPPED_TIERS $v" ;;
    *)        TIER_ENV="$TIER_ENV $v=1" ;;
  esac
done

echo "post-build sweeps -> $OUT (width $MAX_JOBS)"
echo "  tiers enabled :${TIER_ENV:- (none discovered)}"
if [ -n "$SKIPPED_TIERS" ]; then
  echo "  tiers SKIPPED :$SKIPPED_TIERS"
  echo "                  (timing-sensitive; a parallel sweep cannot judge"
  echo "                   them — they need a serial pass. NOT covered here.)"
fi
for f in "$REPO_ROOT"/cogwheel/tests/test_lensing_*.py; do
  base="$(basename "$f" .py)"
  if tail -2 "$OUT/$base.log" 2>/dev/null | grep -qE "[0-9]+ passed"; then
    echo "  skip $base (already green)"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 5; done
  cache="$OUT/numba_cache_$base"
  mkdir -p "$cache"
  (
    cd "$REPO_ROOT" || exit 1
    env $TIER_ENV NUMBA_CACHE_DIR="$cache" \
      OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      NUMBA_NUM_THREADS=4 \
      "$RUNPY" -m pytest "$f" -q -p no:cacheprovider "$@" \
      > "$OUT/$base.log" 2>&1
  ) &
done
# Self-emitting progress: one beat per 3 min on stdout, so ANY observer
# (a Monitor, a tail, a human) sees advancement without instrumenting.
( while true; do sleep 180; n=0; for log in "$OUT"/*.log; do [ -f "$log" ] && tail -2 "$log" 2>/dev/null | grep -qE "[0-9]+ (passed|failed|error)" && n=$((n+1)); done; echo "PROGRESS: $n files tallied ($(date +%H:%M))"; done ) &
beat_pid=$!
fail=0
while [ "$(jobs -rp | wc -l)" -gt 1 ]; do wait -n || fail=1; done
kill $beat_pid 2>/dev/null
echo
echo '=== per-file tallies ==='
for log in "$OUT"/*.log; do
  printf '%-42s %s\n' "$(basename "$log" .log)" \
    "$(tail -2 "$log" | grep -E 'passed|failed|error' | tail -1)"
done
exit $fail
