#!/usr/bin/env bash
# Post-build slow sweeps — the driver's ONE command after every build
# commit (owner mandate 2026-07-21: builds are FAST; slow sweeps are
# POST-BUILD parallel jobs).  Fans one pytest process per lensing test
# file with the slow tiers enabled and a PER-PROCESS numba cache dir
# (concurrent processes racing one shared cache segfault — measured),
# then prints a per-file tally table.  Usage:
#   .claude/sdk/post_build_sweeps.sh [extra pytest args]
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNPY="$REPO_ROOT/.claude/sdk/run_py.sh"
OUT="/tmp/post_build_sweeps_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "post-build sweeps -> $OUT (slow tiers ON, per-process numba cache)"
pids=()
for f in "$REPO_ROOT"/cogwheel/tests/test_lensing_*.py; do
  base="$(basename "$f" .py)"
  cache="$OUT/numba_cache_$base"
  mkdir -p "$cache"
  ( cd "$REPO_ROOT" &&     COGWHEEL_BRUTE_ACCURACY=1 NUMBA_CACHE_DIR="$cache"     "$RUNPY" -m pytest "$f" -q -p no:cacheprovider "$@"     > "$OUT/$base.log" 2>&1 ) &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo; echo '=== per-file tallies ==='
for log in "$OUT"/*.log; do
  printf '%-42s %s
' "$(basename "$log" .log)"     "$(tail -2 "$log" | grep -E 'passed|failed|error' | tail -1)"
done
exit $fail
