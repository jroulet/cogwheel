---
section: Backlog
---

- **THE FAST FULL-SUITE GATE CANNOT COMPLETE — FOUR tests hang in mpmath**
  `[housekeeping]` — diagnosed 2026-08-06 by py-spy, twice: on a gate run
  frozen at 99% for six hours, and on a controlled reproduction that stalled
  at 89% with ALL FOUR xdist workers wedged.

  This is NOT "a test forgot its tier gate" (the framing of the first
  diagnosis, now superseded). Four tests in four different files reach
  `_f_schwinger_mpmath` and never return:

  | test | file:line |
  |---|---|
  | `test_prior_draws_are_finite_or_exact_neg_inf` | `test_lensing_marginalized_likelihood.py:839` |
  | `test_mutation_narrowing_except_turns_neginf_red` | `test_lensing_prior.py:1064` |
  | `test_band_limit_refusal_precedes_coherent_score` | `test_lensing_saddle_likelihood.py:463` |
  | `setUpClass` (whole class dies) | `test_lensing_wedge_dd_arclength.py:119` |

  All four sit in `mpmath.quad` summation under
  `_raw_integral_mp` -> `_f_schwinger_mpmath` -> `f_schwinger`
  (`cogwheel/lensing/chang_refsdal/_schwinger.py:845/866/940`), reached
  variously through `_saddle_grid`, `_positive_parity_grid` and
  `Posterior.lnposterior` -> `_evaluate_envelope` -> `_exact_total`.

  Because four independent tests land there, the defect is in `f_schwinger`'s
  ROUTING, not in any one fixture: some parameter regime these tests naturally
  produce selects arbitrary-precision quadrature, which is unbounded here.

  ## FIXED so far (one of five)

  `test_lensing_surrogate_census.py::LnlTierTestCase::
  test_real_likelihood_tiers_within_bars` now carries `@_TRAIN_TIER_SKIP` (the
  file's existing `COGWHEEL_TRAIN_TIER` gate at :535). Applied per-METHOD: its
  two siblings are cheap and stay in the fast tier. VERIFIED: that class went
  from unbounded to 2 passed / 1 skipped in 3.70 s.

  ## Why this is the highest-priority item in the repo

  The tree-wide fast gate is the COMMIT PRECONDITION for every SDK build. On
  2026-08-06 it hit its own 3600 s timeout at ~88% and STRANDED the
  interior-wedge build, which had already passed Inspector and Professor. So
  this is not merely wasted CPU: it blocks shipping.

  It also means "full suite green" has not been established for some time.
  The post-build tally for the Born residual chart build (`849e580`, shipped
  2026-08-04) was never obtained — the gate meant to verify it is the run
  killed here. See [[lensing_serving_ladder_guards_are_red]] for 11 real
  failures that went unnoticed behind this.

  ## Work, in order

  1. Install `pytest-timeout` (NOT currently available — `--timeout` is an
     unrecognized argument) and add a per-test timeout to
     `.claude/sdk/run_full_suite.sh` AND to the SDK's tree gate. Highest
     value: an unbounded test then fails LOUDLY and NAMES ITSELF instead of
     needing a py-spy autopsy. The gate self-emits `[beat] n/N` on progress,
     which is exactly why hours of no beats read as "still running".
  2. Establish which condition in `f_schwinger` (`_schwinger.py:940`) selects
     `_f_schwinger_mpmath` over the fast branch, and why these four cross it.
     The DD product cap and the mpmath ceiling are different thresholds and
     may disagree.
  3. Fix at the routing level if the regime is legitimate but the
     implementation is unbounded; otherwise shrink the four fixtures. Prefer
     shrinking a fixture over tier-gating it: `setUpClass` of a whole class
     and a `prior draws are finite` smoke check are fast-tier guarantees, and
     a test moved behind an opt-in env var stops guarding anything.

  ACCEPTANCE: `run_full_suite.sh` completes end to end and reports a tally;
  no fast-tier test enters `_f_schwinger_mpmath` (assert it directly — patch
  the symbol and fail if called during the fast tier).
