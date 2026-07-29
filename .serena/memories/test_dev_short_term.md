# Test Dev Short-Term Observations

## Build 8f (F028) select_branch routing — test_lensing_schwinger.py
- WP1/WP2 gave `_positive_parity_grid` and `_saddle_grid` an above-ceiling
  geometric branch routed through the shared `select_branch`. Added 8 new
  classes: SelectBranchOneHome, F028GeometricServe, BelowCeilingByteIdentity,
  SaddleServeBoundaryInvariance, DeltaMinComputedAtMostOnce,
  AboveCeilingWaveThreeOutcome, SelectBranchSelfFalsification (+ shared base
  `_SelectBranchRoutingTestCase`). Full file 48 tests green in 3m18s.
- ONE-HOME recipe: recompute predicate args from PUBLIC helpers
  (`macro_matrix` -> `_real_delay_min_separation` -> for positive parity
  `cancellation_exponent(w,y,g,k)` which EQUALS the grid's `w*y_prime_norm`;
  saddle passes `math.inf`), then OBSERVE the grid's routing by what scalar
  `F_op` serves: served==geometric_amplification(bit-eq)->'geometric', any
  other value or SchwingerCertificationError->'wave', geometry.LensDomainError
  (census guard on the geometric handoff)->'geometric'. Assert node-for-node
  equality + both labels non-vacuous.
- ONEHOME sources MUST be OFF-AXIS: on-axis y=(a,0) gives mirror-degenerate
  Fermat delays -> delta_min=0 -> resolution leg dead everywhere, geometric
  outcome never appears. Used fixed unit dir (0.8,0.6) scaled to each |y|.
- F028 exact-serve is a ROUTING pin (served == geometric_amplification, ==);
  accuracy anchored SEPARATELY below/at ceiling vs the exact `F_op` quadrature
  (independent oracle, tol 1e-4, measured worst ~4e-6). Do NOT assert geom vs
  quadrature above w=60 (F_op IS the arm there; diff identically 0).
- RefusalAboveCeilingTestCase needed NO re-pointing: all ABOVE_CEILING
  fixtures (g in {0.47,0.49}, |y| small) measured ce<48 -> still 'wave'.
- Above-ceiling refusal message is y-INDEPENDENT (f_schwinger ceiling guard
  fires before y work) -> reproduce authentic message by a direct
  f_schwinger(w, any_valid_y_eig, gp) call; full str matches F_op's.
- delta_min spy: mock.patch.object(operator,'_real_delay_min_separation',
  side_effect=original); grid computes it ONCE before the node pre-pass, so a
  refusing above-ceiling grid still counts 1 (wrap F_op_grid in try/except).
  All-below grids -> 0. Verified pos & saddle.
- Byte-freeze: 5 both-parity configs x {5,40,59}, frozen as float.hex()
  literals; HEAD-worktree vs post-build verified IDENTICAL before baking.

## Build 8f continuation (F028) — verification-only pass
- Predecessor actually FINISHED the suite before budget end; its note was
  accurate. Re-verified whole file: 48 passed in 197.8s, 0 skips/xfails.
- Blast-radius audit for this file's ~1237/1255/1277 lands in
  RefusalAboveCeilingTestCase: its ABOVE_CEILING fixtures (g in {0.47,0.49},
  small |y|) are genuinely WAVE-routed (ce<L_MAX), so spec's "keep
  arm/refusal" branch applies -> correctly UNCHANGED, not a silent flip.
  Green run is the proof: geometric serve would break served==arm.
- Diff = 558 ins / 2 del; the 2 dels are just the import line expanding to
  add select_branch/geometric_amplification/etc. No assertion weakened. No
  edits needed this pass.
