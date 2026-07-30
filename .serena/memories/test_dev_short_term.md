# Test Dev Short-Term Observations

## Build (WP1 F041 _make_arc arc-orientation) — test_lensing_surrogate_training.py (shard 2)
- Owned ONLY this file. (A) Fixed 3 stale `_find_cusps` callers now that
  gamma/branch are REQUIRED kw-only: L1006 `_wp3_fixoff_left_arc` ->
  `gamma=gamma, branch=branch`; L1660/1661 astroid-guard test -> both get
  `gamma=float(gamma), branch=1` (frozen + widened). L1947 `shifted` wrapper
  forwards `**kwargs` -> untouched, OK. NO removed-const refs in this file
  (grep _CUSP_SPEED_REL_FRAC/_CLOUD_MARGIN_FRAC/_PROBE_ETA = none); the
  _CUSP_WIDTH_SAFETY/_CUSP_MIN_HALFWIDTH/_SADDLE_* imports are all LIVE.
  Nothing to delete.
- ALL 3 callers sit inside @_TRAIN_TIER_SKIP classes (SaddleTubeTailTestCase
  via _wp3_fixture; AstroidByteIdentityTestCase). Verified: called
  `_wp3_fixoff_left_arc` in ISOLATION (pure geometry, no engine) -> runs;
  ran the 2 astroid-guard tests under COGWHEEL_TRAIN_TIER=1 -> 2 passed 3.3s.
- (B) Authored StableGammaBandsF041TestCase (3 tests, NOT skipped, fast-tier)
  + StableGammaBandsF041SelfFalsificationTestCase (2 tests). Full fast-tier
  file: 9 passed, 36 skipped, 4.8s. F041-only: 5 passed 5s.
- MEASURED (post-fix): stable_gamma_bands((0.01,0.30),+1,200,0.02) = ONE band
  (0.01,0.30), narcs=2 (NOT 4 — astroid yields 2 fold arcs after cusp-window
  exclusion), dropped=[]. Every arc image_count=4, inward_sign=-1. one
  detect=0.015s, main sweep=0.043s.
- REGRESSION WITNESS proven vs HEAD (git worktree add /tmp/f041_head HEAD):
  pre-fix HEAD -> dropped=[(0.01,0.028125),(0.04625,0.064375)], 4 bands. So
  assertion `dropped==[]` genuinely goes RED on HEAD. Spec's "2 bands build
  ZERO arcs" = these dropped slivers.
- SELF-FALSIFICATION (fast, no engine): patch training.detect_caustic_structure
  to strip arcs for gamma<0.05 (dataclasses.replace(s,arcs=())) — reproduces
  BOTH limbs: dropped=[(0.04625,0.064375)] AND a zero-arc band (arcs=[0,2,2,2]).
  stable_gamma_bands->band_caustic_structure->detect_caustic_structure all
  module-global, mock.patch.object(training,...) works; capture real ref
  before patch to avoid recursion.
- Assertion-3 gotcha: main sweep is ONE band lo=0.01<0.1, so "bands with
  lo>=0.1" filter is empty. Realized label stability by (i) a separate
  stable_gamma_bands((0.1,0.30)) per-arc image_count/inward_sign check AND
  (ii) cross-gamma per-index label consistency over detect structs at
  {0.1,0.2,0.3,0.9}. detect sig = (gamma, parity, *, n_samples=200).
- NOTE: cwd resolves to /home/tejaswi/Work/cogwheel-claude-dev (serena project
  root). Edits + prod WP1 fix both live there. Python:
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin.

## Build 8i (WP1 F041 _make_arc arc-orientation) — test_lensing_surrogate.py
- Shard owned ONLY this file. Single-line fix: line 1068 in
  ClosedFormCuspAngleTestCase.test_cusp_magnitude_varies_with_gamma —
  `_find_cusps(thetas, speed, periodic=True)` -> added `gamma=gamma, branch=1`
  (branch=1 matches the loop's _branch_speed_profile(gamma,1,...)). This
  unblocked the RUN abort that tripped the ~688 anti-vacuity teardown.
- NO `shifted`/real_find_cusps wrapper in THIS file (that's in
  test_lensing_surrogate_training.py :1947). Only ONE live _find_cusps call
  here (the 1006 hit is a docstring). NO removed-const refs
  (_PROBE_ETA/_CLOUD_MARGIN_FRAC/_CUSP_SPEED_REL_FRAC) — nothing to delete.
- WP1 changed _make_arc arc-orientation guard; grep of THIS file for
  _make_arc/inward_sign/arc_orientation/_tube_normal = none (two "orientation"
  hits are beta-shear docstrings). No pin, no rot.
- ClosedFormCuspAngleTestCase 4 passed 3.9s; FULL file 51 passed, 1 skipped in
  226s. No production edits.

## Build (WP1/2/3 exact-geometry housekeeping) — test_lensing_caustic_cusps.py
- Assigned the deletion/re-baseline housekeeping. File is UNTRACKED (authored
  fresh THIS build across shards) so it already satisfies the spec by
  construction: NO st._PROBE_ETA/_CLOUD_MARGIN_FRAC/_CUSP_SPEED_REL_FRAC pins,
  NO astroid byte-identity test (only CuspWindowByteIdentityTestCase = WIDTH
  byte-identity survives), every live st._find_cusps(...) already passes
  gamma=/branch=. 28 passed in 63s. Made NO edits (verification + audit only).
- KEY: HEAD is the PRE-change baseline (uncommitted WP). HEAD's _find_cusps is
  the OLD self-contained version (no gamma/branch, uses _CUSP_SPEED_REL_FRAC)
  and HEAD still HAS the 3 removed consts (lines 108/129/155). So
  _head_find_cusps() AST-extract oracle works correctly; live st module has
  the consts REMOVED (grep cogwheel/lensing empty). INCUMBENT_CLOUD_MARGIN_FRAC
  =0.10 in the file is a LOCAL successor-gate constant, NOT a prod pin — fine.
- BACKWARD-COMPAT ROT (REPORTED, other-run-owned — NOT edited):
  * test_lensing_surrogate.py:1068 _find_cusps(...,periodic=True) missing
    gamma/branch -> TypeError.
  * test_lensing_surrogate_training.py:1006, 1660, 1661 _find_cusps(...) missing
    gamma/branch -> TypeError. :1947 real_find_cusps(...,**kwargs) wrapper
    forwards kwargs -> likely OK.
  * test_lensing_exterior_admission.py:1597,1607,1613,1657,1669,1684,1737
    reference st._CLOUD_MARGIN_FRAC (removed) -> AttributeError. Pin the RETIRED
    cloud+margin mechanism (behavior genuinely gone) -> PROPOSE delete/replace;
    successor gate is InteriorAdmissionMarginRemovalTestCase in the cusps suite.


## Build 8g-cont (WP1/2/3 exact geometry) — EXTENDED test_lensing_caustic_cusps.py
- Added 5 classes (+5 self-falsifiers, +2 plots) for serve-alignment health,
  closed-form inradius, foot-of-normal pin(a), stable-band sliver pin(b),
  interior-admission margin-removal pin(c). Full file 28 passed in 63s.
- INWARD_SIGN HEALTH: reproduce _make_arc's chosen theta by replaying its
  frac loop (0.5,0.35,0.65,0.2,0.8) over arc.[theta_lo,theta_hi] with the same
  |dot|>0.1 floor + _tube_normal serve normal. |dot| measured min 0.298 (pos
  g=0.2); branch=-1 saddle edges = 1.0. sign(dot)==inward_sign always. Both
  parities, both saddle branches.
- INRADIUS: _caustic_inradius(g,+1,200)==g to ~1e-16; independent det J=0
  closed form (s=1/r^2=g*cos2f+sqrt(1-g^2 sin^2 2f); y=((a-s)x1,(b-s)x2),
  a=1-g,b=1+g; min|y|) agrees to ~2e-10 (<1e-9 rtol). MY FIRST oracle was
  WRONG: used x=(cosf,sinf) i.e. r=1; correct is r=1/sqrt(s). Verified against
  shipped critical_point.source before trusting. DEVIATION: closest approach is
  a SMOOTH waist (argmin f~0.6-2.5), NOT a cusp — axis cusps sit at |y|=2g/
  sqrt(1±g) > g. encloses_origin=True (winding pin).
- PIN(a) foot-of-normal: main bands (.25,.35)(.45,.55)(.65,.75)(.85,.95) +
  (.155,.3) give eta_max>0.5*rmin = FALSE (brief headline OK). DEVIATION:
  (.0825,.155) = TRUE (rmin=0.059, 0.5rmin=0.030<0.05 — small astroid tight
  curvature, guard correctly fires). (.0281,.0462)=0 arcs; (.0644,.0825)=
  CausticTopologyError [0,2,2] — no band-wide arc to test.
- PIN(b) stable_gamma_bands((.01,.30),+1) drops EXACTLY ONE sliver
  (0.064375,0.0825) [bisection-exact, NOT brief's 0.0644], not zero. Cause:
  astroid-onset metamorphosis arc count 0->2, width 0.018<min_width 0.02.
- PIN(c) margin removal: new exact-distance admits() ⊇ incumbent(cloud+0.10)
  over 9x9 tile grid; 3-4 flips/band all old-refuse->new-admit; each flip's
  INDEPENDENT dense-cloud(40001) clearance in [eta_max, 1.1*eta_max). Incumbent
  oracle = faithful HEAD admits() transcription (git show HEAD verified). NOT
  vacuously equal — asserts flips>0.
- BACKWARD-COMPAT ROT (REPORTED, NOT edited — other-run-owned suite): WP
  DELETED _CLOUD_MARGIN_FRAC from production (hasattr False). This ERRORS
  test_lensing_exterior_admission.py::CloudMarginTestCase 4/5 tests:
    * test_margin_zero_admits_the_genuine_false_admit (:1597 mock.patch.object
      missing attr -> AttributeError)
    * test_margin_frac_at_least_ten_percent (:1607 st._CLOUD_MARGIN_FRAC)
    * test_margin_width_exceeds_worst_measured_slop (:1613)
    * test_exterior_admitted_set_unchanged_under_margin (:1657/:1669)
  test_comfortably_interior_tile_still_admits survives (admits() only). These
  pin the RETIRED cloud+margin mechanism — behavior genuinely gone; PROPOSE
  delete/replace, my InteriorAdmissionMarginRemovalTestCase is the successor
  gate. Other refs OK: _caustic_inradius 2-tuple unpack + _interior_admission
  4-arg sig unchanged in exterior_windows/ppgo_bandsplit.


## Build 8g (WP1/2/3) cusp/fold refactor — NEW test_lensing_caustic_cusps.py
- New suite (13 tests, 4.2s green) for surrogate_training._find_cusps
  relocation + served-image-count. All 3 live gates proven red under
  mutation (shifted cusp / scaled delta / flipped inward_sign).
- SPEC1 (astroid cusp = analytic root): astroid cusps at gamma in
  {0.05,0.2,0.4,0.7} sit EXACTLY on the n=200 grid (pi/2=50*2pi/200 etc.)
  so relocation keeps grid value; axis-coincidence machine-exact (0.0),
  speed/peak ~4e-16 << 1e-6. Oracle = closed-form {0,pi/2,pi,3pi/2}.
- SPEC2 (window byte-identity): HEAD _find_cusps is fully self-contained
  (np + _CUSP_SPEED_REL_FRAC/_CUSP_WIDTH_SAFETY/_CUSP_MIN_HALFWIDTH). AST-
  extract the FunctionDef from `git show HEAD:...surrogate_training.py`,
  literal_eval the 3 consts, exec in minimal namespace. delta byte-equal
  (assertEqual on float); HEAD sig has NO gamma/branch (call w/o them).
  NEW inline window_dip_frac=0.2 == HEAD _CUSP_SPEED_REL_FRAC=0.2.
- SPEC3 (Professor served=4/opp=2): STRICT gate holds for positive
  {0.2,0.4,0.7,0.9} + saddle 1.2. At saddle 1.5 served=4 but branch=-1
  opposite raises LensDomainError (census defect) -> weaker gate
  served==4 AND opp!=4 (in {2,'LensDomainError'}). At saddle 2.0 eta_max
  =0.05 OVERSHOOTS pinched branch=1 lobe (r_min~0.194): served drops to 2
  but eta=0.02 restores 4 -> EtaOvershootBoundaryTestCase witnesses it's
  overshoot not parity error. DOCUMENTED SPEC-3 DEVIATION, reported.
  _real_image_count helper returns int or 'LensDomainError' string.
- BACKWARD-COMPAT ROT (reported, NOT edited — other-run-owned suites):
  WP1/2 made gamma+branch REQUIRED keyword-only on _find_cusps. Broken
  live calls (TypeError: missing gamma/branch):
    * test_lensing_surrogate.py:1067-68 (add gamma=gamma, branch=1)
    * test_lensing_surrogate_training.py:1006 _wp3_fixoff_left_arc
      (add gamma=gamma, branch=branch)
    * test_lensing_surrogate_training.py:1660,1661 (add gamma=float(gamma),
      branch=1)
  The :1947 patch-wrapper shifted(...,**kwargs) forwards gamma/branch — OK.
  _CUSP_SPEED_REL_FRAC removed from prod but only referenced in MY HEAD
  oracle — no other test imports it.

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
