# Build: refresh tiling cost estimate at the order-16 diffractive certificate

## Mission

Re-run the demand-sized tiling plan + campaign cost estimate (order-7a step 2)
at the NEW diffractive certificate. The prior estimate
(`tiling_plan_and_cost_7a2.json`, 2026-08-19) was built on the PRE-FIX
certificate (order 8, first-breach search absent, over-certifying up to +78%
in w). Since then two things changed the demand map, and both shrink it:

1. The first-breach certificate fix (ec195b7): engine_residual fell from
   44.62% (10k) to 37.27% (3k) at order 8.
2. The order raise 8 -> 16 (530e397): engine_residual fell further to
   32.80% (3k); the exterior closes 100% any-coverage at order 16.

The old estimate carried an OPEN ESCALATION: `exterior:+1` was 54% of planned
nodes (> 40% limit). That escalation was the "82% of budget on the astroid
exterior" hazard re-measured — and the certificate fix targets exactly that
population. The refresh must re-answer: is exterior still the dominant cost,
or has the analytics-first doctrine now structurally prevented it?

## Deliverables

1. Re-run `scripts/tiling_plan.py` (engine-free; refreshes the 10k serve-route
   census internally at HEAD = order-16 certificate) -> new combined plan+cost
   JSON. The census refresh MUST use the SHIPPED production predicates as-is
   (mirror-fidelity — no re-typed gates), so the order-16 default is picked up
   automatically from `_diffractive._DEFAULT_MAX_ORDER`.
2. Compare against `tiling_plan_and_cost_7a2.json` (the old estimate):
   report the new `engine_residual` share, per-region node/call counts,
   totals, and the escalation verdict. State explicitly whether the
   `exterior:+1` 54% -> ~? escalation cleared, and WHY (fewer residual draws
   classified `engine_residual` -> fewer/cheaper exterior tiles).
3. Reconcile the plan cross-checks (vs `_self_estimate`, vs `tiling_census`)
   and the `residual_vs_ledger` ratio against the new 32.80% measurement.
4. Write the new estimate to `.claude/handoff/tiling_plan_and_cost_order16.json`
   (or a name the driver approves) and record the deltas in the campaign todo
   fragment `todo.d/lensing_training_campaign.md` step-2 status.

## Measured facts (verified at 530e397, engine-free)

- order-16 3k census (seed 0, 20-1024 Hz, n_freq 128): engine_residual
  984/3000 = 32.80%; diffractive_analytic 455; born_analytic 119;
  ppgo_above_ceiling 504; saddle_c3 400; wave_refused 67.
  NOTE the 3k census undersamples vs the tiling plan's own 10k refresh —
  the build's 10k numbers are authoritative and may differ by ~1-2 pct pts.
- order scan (40 draws/region): exterior any-coverage 75/95/100/100% at
  orders 8/12/16/24; whole-band 10/25/30/50%. wedge_interior any 10/17.5/
  22.5/40%. The series is convergent — reach is a monotone function of order.
- engine-honesty: worst rel-err at the served ceiling 1.000e-4 (within the
  1% estimator allowance); gamma=0.1 ceiling 40.9 at order 16.

## Acceptance

- The refreshed plan is a THIN CALLER re-run: `tiling_plan.py` + the census
  at HEAD, no hand-tuned numbers. The `engine_residual` share it reports
  should be within ~1-2 pct pts of 32.80% (3k) and clearly below the 42.06%
  the old estimate measured (10k pre-fix).
- Escalation verdict re-derived and justified against the new counts.
- No chart training, no campaign run — plan-only (step 2 deliverable).

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build tests FAST.
- Spec/TODO workflow applies: this UPDATES the step-2 status of the open
  campaign fragment; no new spec surface beyond the numbers it records.
- The census is ~45 min at order 8, slower at order 16 — run it ONCE at
  10k with the shared warm numba cache (`NUMBA_CACHE_DIR=/tmp/numba_census_shared`),
  do NOT hand-roll a parallel census (serial, progress-tracked, is fine and
  reliable). Report progress every 250 draws.
