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
echo "post-build sweeps -> $OUT (slow tiers ON, width $MAX_JOBS)"
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
    COGWHEEL_BRUTE_ACCURACY=1 NUMBA_CACHE_DIR="$cache" \
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
