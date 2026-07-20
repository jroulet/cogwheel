# Professor short-term — Build 8c inference review (2026-07-20)

## What I reviewed
Domain correctness of the in-progress (UNCOMMITTED) Build-8c multi-chart lens
amplification surrogate + census tool. Worktree cogwheel-claude-dev, env
cogwheel-newlal (py3.10). Authoritative plan: build8c_plan_approved.json (11
domain test descriptions).

## Fast tests run
`pytest cogwheel/tests/test_lensing_surrogate.py` = **33 passed, 1 skipped**
in 296s. Skip = TimingSmokeTestCase (opt-in COGWHEEL_RUN_TIMING_SMOKE, perf
only, not physics). Slow part = dd-Schwinger saddle training fixture
(lru_cached, front-loaded).

## Physics verdicts (all CORRECT)
- Census tier tolerances EXACTLY my ratified rulings: CROWN_LNL_TOL=0.05
  (Professor override of unreachable 0.01; dlnL~eps*SNR^2 floors ~0.04),
  STRONG_SADDLE=0.1, RESCUED=1.5==RB_ATOL. assign_tier uses only certified
  (gamma,eta) axes (theta excluded per F017); near-caustic + strong-shear +
  macro-saddle all -> strong_saddle. Verified by direct call.
- classify_fallthrough is MECE, calls surrogate's OWN guard predicates (one
  source of truth), Q7 arc-projection -> out-of-box (NOT cusp-window) is
  explicitly coded + documented. 5 cats: gamma-guard/dropped-sliver/
  cusp-window/refusal-ball/out-of-box.
- lnL accuracy test uses the physically-right envelope-scaling gate
  dlnL <= 1.5*eps_dense*|lnL_exact| (i.e. eps*SNR^2), + 0.5-nat fixture
  budget ceiling; 0.01/0.05 recorded as asymptotic targets. Passed.
- Mass-sheet INS-8a-001: nonzero_kappa_never_served PASSED. Refusal
  preservation, oracle-independence (F002), crown byte-identity (default
  None), F010 mutation, chart-selection determinism + neg-theta wedge
  unwrap, single-npz serialization+provenance all PASSED.
- Registration (plan TEST10): pipeline_graph resolves
  lens_amplification_surrogate -> producer train_lens_surrogate.py::main,
  registry_path=yes, consumers include LensedRelativeBinningLikelihood,
  LensedMarginalizedExtrinsicLikelihood, surrogate_census.run. Verified via CLI.

## CONCERN (coverage gap, not a physics error)
- The two DESIGN-FALSIFIABLE tests are ABSENT from pytest: tube-beats-raw
  >=3x eps_95 (plan TEST1/task TEST8) and fold-approach slope tube-flat vs
  raw -1/2 (plan TEST2/task TEST9). These justify the sqrt(eta) tube
  coordinate; the central design premise is unverified by automated test.
- Census-tool correctness (plan TEST9: served-fraction/fall-through
  breakdown vs hand-computed, per-chart held-out eps) has NO pytest — only
  the CLI scripts/census_lens_surrogate.py. I verified tier+categorizer
  logic by hand/direct call, but there's no regression guard.
- Registration also has no dedicated pytest (verified manually).
- Build is uncommitted WIP (WP2/WP3 may be unfinished). Heavy full-scale
  training/sampling is operator-deferred.

Verdict issued: CONCERN — everything built is domain-correct with ratified
tolerances, but the design's falsifiable claim + census correctness lack
automated tests. No image viewer available to me; relied on passing
assertions (which encode the tolerances) + code review.
