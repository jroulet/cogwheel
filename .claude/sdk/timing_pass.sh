#!/usr/bin/env bash
# Serial timing tier — the driver's OCCASIONAL command, not a per-build step.
#
# WHY IT IS SEPARATE FROM post_build_sweeps.sh
#
# 1. It is SERIAL by necessity. Every assertion here is a speed comparison;
#    under the sweep's 8-wide contention it would measure the sweep.
# 2. It is EXPENSIVE. Measured 2026-07-30: 44:44 wall clock, because the
#    strict branches time `lnlike_bruteforce` as the reference — ~18 brute
#    evaluations across the sites. Bolting that onto a ~25 min sweep would
#    nearly triple every build's post-step to re-check ratios that do not
#    silently drift.
# 3. Nothing had EVER run it (F052). COGWHEEL_STRICT_TIMING gates four
#    `if _STRICT_TIMING:` branches that tighten assertions rather than skip
#    tests, so the suite always took the loose path and no job set the
#    variable. First run: all four passed.
#
# WHEN TO RUN: before a release, after touching the fast path / relative
# binning / the surrogate serve path, or when a speedup claim is in doubt.
#
# Usage: .claude/sdk/timing_pass.sh [extra pytest args]
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNPY="$REPO_ROOT/.claude/sdk/run_py.sh"
OUT="${TIMING_OUT:-/tmp/timing_pass_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# The classes carrying timing assertions. Kept explicit: this list is short,
# and a wrong -k here silently measures nothing.
K="CausticSearchPreservationTestCase or FewMsTimingTestCase \
or RatioTimingTestCase or TimingSmokeTestCase"

FILES="
$REPO_ROOT/cogwheel/tests/test_lensing_batched_operator.py
$REPO_ROOT/cogwheel/tests/test_lensing_fast_path.py
$REPO_ROOT/cogwheel/tests/test_lensing_ratio_layer.py
$REPO_ROOT/cogwheel/tests/test_lensing_surrogate.py
"

echo "serial timing pass -> $OUT"
echo "  tiers: COGWHEEL_STRICT_TIMING=1 COGWHEEL_RUN_TIMING_SMOKE=1"
echo "  expect ~45 min; the brute-force reference calls dominate"
echo "  NOTE absolute wall-clock numbers are machine-dependent; the SPEEDUP"
echo "       ratios are the portable claims and are what is gated."

# Progress beats: with `-v` each test prints on completion, so a stalled run
# is distinguishable from a slow one. (With `-q` a long single test emits
# nothing, which once read as a stall — health is CPU, not log growth.)
cd "$REPO_ROOT" || exit 1
COGWHEEL_STRICT_TIMING=1 COGWHEEL_RUN_TIMING_SMOKE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$RUNPY" -m pytest $FILES -v -p no:cacheprovider -k "$K" "$@" \
  2>&1 | tee "$OUT/timing.log"
rc=${PIPESTATUS[0]}

echo
echo '=== measured timings (the point of this pass) ==='
grep -aE '\[(FewMsTiming|TimingSmoke|RatioTiming)\]|speedup|speed-up' \
  "$OUT/timing.log" || echo '  (none printed)'
echo
if [ "$rc" -eq 0 ]; then
  echo "TIMING PASS: GREEN"
else
  echo "TIMING PASS: RED — check whether the box was busy before blaming code:"
  grep -aE '^(FAILED|ERROR)' "$OUT/timing.log" | head
  echo "  ps -eo pcpu,etime,args --sort=-pcpu | head"
fi
exit "$rc"
