# Test Dev Short-Term Observations

## Build (F038/F039) caustic derivatives — test_lensing_caustic_derivatives.py
- Added 3 TestCases to the (untracked, never-run) suite + fixed a pre-existing
  false-positive in OracleIndependenceTestCase. Full file 20 passed in 4.7s;
  sibling test_lensing_geometry.py 13 passed (no regression, my change is
  test-only, one untracked file).
- STAGE-1 curve validation: iterate real_cases() (F038 set), compare mpmath
  oracle _oracle_y_component to critical_point(...).source[i]. MUST mirror both
  critical_point quirks: clamp slightly-neg saddle discriminant to 0, ignore
  branch at positive parity. Headline rel-err 5.144e-15 over 76
  relative-dominated points == driver's 5.14e-15 (strong independence proof).
- STAGE-1 gate is TWO-PART: near-axial (theta~0.02) is PURE float64
  cancellation (abs err 4.3e-18 but rel 5.3e-12) because shipped Ax-x/r^2 and
  oracle p*r*T are algebraically identical (mpmath diff 4e-50) but differ in
  float64. So: per-point MIXED tol (atol=1e-13,rtol=1e-12) covers all points +
  headline rtol=1e-13 gate ONLY on |shipped|>0.1 (the ATOL/RTOL crossover, not
  an ad-hoc conditioning floor — |expected| is a gapless continuum).
- Self-falsification for stage-1 MUST pick a well-conditioned point
  (|source|>0.1 floor) or it lands in the near-axial regime and the wrong-curve
  mutation hides under cancellation: used gamma=0.99,theta=1.3,comp=1
  (|source|~17.66).
- OracleIndependence false-positive: _FORBIDDEN_ORACLE_NAMES had
  'y_prime'/'y_double_prime' which collide with oracle_derivatives's OWN local
  accumulators (they're oracle outputs, not module cascade symbols) — removed.
- POSITIVE-PARITY branch=-1: under simplefilter('error',RuntimeWarning) call
  caustic_derivatives/speed/curvature_radius branch=-1 vs +1; assert_array_equal
  (exactly 0.0 — positive parity ignores branch, Prof Q5). Positive control:
  np.sqrt([-1.0]) must raise under the armed filter.
- FOLD: positive-parity only (filter |gamma|<1-kappa; (0.9,0.3) is a macro
  saddle -> LensDomainError, excluded). eps=1e-3, thetas away from astroid
  cusps 0/pi/2/pi/3pi/2 so the merging pair resolves. n_plus>n_minus on +d
  side; d unit (|norm-1|<=1e-12); invariant under soft_axis sign flip
  (mock.patch critical_point returning ._replace(soft_axis=-soft_axis)).
  Diagnostic scatter -> output/fold_opening_direction_image_counts.png.

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
