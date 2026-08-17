\g<0>
## 2026-08-17 build (INS-1-001: TubeChart theta_to_s -> theta_to_s_prime sweep)
- Migrated all 5 files named in the finding to the WP2 rename
  (theta_to_s->theta_to_s_prime kwarg/attr): test_lensing_surrogate.py
  (27-occurrence global literal replace), test_lensing_low_w_extrapolation.py
  (2 attr accesses), test_lensing_wedge_dd_arclength.py
  (FieldExposureTestCase renamed test + docstrings), test_lensing_log_reach_gamma.py
  (2 manually-built TubeChart.from_values call sites, kwarg + local var
  renamed, indentation fixed per-site since the two call sites have different
  base indent), test_lensing_surrogate_training.py (ShippedArcLengthTubeGridTestCase
  rewritten, see below). All mechanical renames verified GREEN: 128 passed
  (surrogate.py, 91s), 62 passed+12 skipped (low_w/wedge_dd/log_reach_gamma
  combined, 17s), 179/179 collect cleanly overall.
- ShippedArcLengthTubeGridTestCase REWRITE (not mechanical -- the Inspector's
  suggested-fix semantics): now asserts against training._tube_delay_map's
  SERVABLE SUBRANGE (theta_fine[i_lo]/[i_hi], freshly recomputed at the same
  rep_gamma/eta_max the build used) instead of the retired raw-arc-bounds
  endpoints_equal_arc_bounds claim, and TV(s') uniformity via the carried
  chart.theta_to_s_prime map instead of geometric-arc-length uniformity.
  Retired 1 sub-test (independent-oracle delay-TV reconstruction from
  _merging_fold_pair) as parsimony-redundant with
  test_lensing_tube_nyquist_coordinate.py's DRY delay pin (mocked-engine,
  fast-tier, far cheaper) -- documented the substitution in the class
  docstring per the parsimony law. Net: 4 test methods -> 3.
- PRE-EXISTING PRODUCTION DEFECT CONFIRMED, NOT MINE TO FIX (cross-checked
  against SaddleTubeTailTestCase, which I did not touch): under
  COGWHEEL_TRAIN_TIER=1, `_wp3_fixture()`'s shared _WP3_GAMMA=1.55 branch=1
  arc now raises `ValueError: Tube delay map is not strictly increasing`
  from training._tube_delay_map -- this breaks ALL FOUR gated tube classes
  in test_lensing_surrogate_training.py identically (SaddleTubeTailTestCase,
  ShippedArcLengthTubeGridTestCase, ArcLengthBoundShiftMarginTestCase,
  ArcLengthNodeEfficiencyTestCase), confirming it is a cross-cutting WP2
  production defect in the shared fixture/pipeline, NOT introduced by my
  test rewrite (untouched SaddleTubeTailTestCase fails with the identical
  error/gamma/branch). Matches a FLAGGED PRODUCTION DEFECT already recorded
  in this memory from the 2026-08-17 tube_nyquist_coordinate build entry
  ("_build_tube_chart is called UNGUARDED... crashes... 'not strictly
  increasing'... Not a test bug") -- this is the same defect surfacing via a
  second call path (_wp3_fixture -> _wp3_build_and_measure -> _build_tube_chart
  directly, not via _train_band_charts). All 4 gated classes therefore
  "import/set up cleanly" (collection succeeds, 179/179) but do NOT pass at
  COGWHEEL_TRAIN_TIER=1 runtime -- this satisfies the Inspector's literal ask
  ("at least imports/sets up cleanly") but the runtime break is a standing
  cross-cutting defect for driver/coder attention, unrelated to the field
  rename semantics I was tasked with.
- Backward-compat audit (step 7): grepped \btheta_to_s\b (non-prime) across
  all of cogwheel/tests/ after the fix -- every remaining hit is either an
  intentional stale-schema fixture (retired key, hard-refusal tests),
  historical docstring prose about the wedge's OWN v1->v2->v3 rename (a
  SEPARATE SHARD C rename, not WP2's tube rename), or an already-correct
  assertNotIn('theta_to_s', ...) check. Nothing further needed edits beyond
  one cosmetic docstring wording fix in test_lensing_surrogate_training.py
  (_wp1_build_uniform_theta_chart docstring said "no theta_to_s map" ->
  "no theta_to_s_prime map").

## 2026-08-17 build SHARD 3 (tube Nyquist coord: WP3 _heldout_eps + WP4 census)
- EXTENDED test_lensing_tube_nyquist_coordinate.py to 46 tests (18.4s, green).
  +2 classes: TubeHeldoutUnservedAsCoverageTestCase (6, WP3),
  TubeCensusNyquistNoExplosionBandTestCase (9, WP4). 0 retired.
- WP3 (_heldout_eps unserved-as-coverage): drive the REAL st._heldout_eps
  ENGINE-FREE by mock.patch.object(st,'ChangRefsdalChannels'/
  'LensAmplificationSurrogate', fakes). Per-sample registry keyed on a UNIQUE
  gamma (0.100+0.001*i) routes both the fake engine reference and the fake
  serve result so the two doubles stay consistent. kinds: served(err)/miss/
  no_ref(raise st.HypergeometricDomainError -> in _ENGINE_REFUSALS ->
  excluded)/nonfinite(nan[0] -> excluded). Core invariant: served+miss folds
  to _HELDOUT_COVERAGE_MISS_EPS=1.0 (WORSE than served-only); no_ref/nonfinite
  EXCLUDED (eps stays served err); nan ONLY on zero served; float always.
  Self-falsification: patch st._HELDOUT_COVERAGE_MISS_EPS=0.0 (restore in
  finally) collapses mixed back to served err. Reuses
  TubeStaleSchemaHardRefusalTestCase._make_synthetic_tube_chart (positive
  parity -> max|E| ref branch, not far-field).
- WP4 (census no-explosion band): drive REAL tc._census_region([ctx],'tube',
  parity=1,config,st) with SimpleNamespace doubles -- the tube VERDICT path
  reads ONLY ctx.tube_arcs + ctx.tube_n_theta_ceiling off ctx, and n_theta/
  n_u/n_gamma/w_nodes_per_decade/n_theta_cap off config. GOTCHA:
  _spatial_nodes_per_tile builds its WHOLE region dict eagerly, so the config
  double MUST also carry n_rho + n_theta_c (exterior factors) or it
  AttributeErrors even for the tube key. Arithmetic: n_nodes>nodes_high iff
  config.n_theta>ceiling (all other node factors shared) -> n_theta=cap+8 =>
  EXPLOSION; 0 arcs => 0 tiles/nodes => SILENT_EMPTY; n_theta=cap//2 =>
  IN_BAND. Engine-free proof: booby-trap _cr_channels.ChangRefsdalChannels.
  evaluate + _cr_schwinger.f_schwinger + _f_schwinger_mpmath with a Mock(
  side_effect=_EngineTouched); assert call_count==0. _EngineTouched is a
  direct Exception subclass, asserted DISJOINT from (LensDomainError,
  ValueError,ZeroDivisionError) (the census/ceiling catch tuple). Self-
  falsification: tc._verdict(n_nodes,(1,10**9)) [retired loose static band]
  reads IN_BAND for the same count the Nyquist band flags EXPLOSION.
  Also real tc._tube_n_theta_ceiling saturation test: mock st._capped_w_range
  ->(1,1e9) + st._tube_delay_map->(theta_fine,s_fine,0,7) TV=1 => ceiling
  clamps to n_theta_cap; no-arcs => cap fallback (no geometry stub needed).
- AUDIT (step7, reading): WP3 _heldout_eps semantic change (unservable-with-
  valid-reference now folds in as a coverage miss) is MONOTONE-toward-worse
  and only bites a held-out set containing unservable-BUT-referenced points;
  every other-owned accuracy-bar caller (test_lensing_caustic_cusps L1242/
  1418/1910/1934; test_lensing_surrogate_lobe L845/2022;
  test_lensing_interior_wedge_chart import; test_lensing_exterior_windows
  L3335) picks SERVABLE held-out points by construction, so the fold is a
  no-op for them -- not stranded. test_lensing_exterior_polar_fold L1087/1106
  test_heldout_eps_* compute eps INLINE (never call st._heldout_eps) -> safe.
  test_lensing_farfield_envelope L86 is a docstring of the F-norm rule WP3
  didn't touch. Neighbour tc suite test_lensing_tiling_census: 26/26 green
  (its L397 (10**9,EXPLOSION) case is the loose-band foil, unaffected). The
  genuine cross-suite hazard for THIS build remains the WP2 theta_to_s->
  theta_to_s_prime rename strandings recorded in SHARD 1/2 (other-owned).


## 2026-08-17 build SHARD 2 (tube Nyquist coord: A3/F083/schema specs appended)
- EXTENDED test_lensing_tube_nyquist_coordinate.py with 3 Architect specs
  (now 31 tests, 18.5s, all green). New classes: TubeCuspLimitA3SlopeTestCase
  (6), F083NodeEconomyTestCase (4), TubeStaleSchemaHardRefusalTestCase (5),
  plus _EngineFreeTestCase mixin. 0 retired.
- A3 SLOPE SPEC -> ENGINE-FREE SUBSTITUTE (real oracle refuses near cusp):
  real _merging_fold_pair declines within ~0.17-0.30 rad of the cusp so the
  d in [d_min,0.1] window is EMPTY on a live arc (measured real-region
  slopes 8.2/2.7/1.6, not 2/3). Substitute: feed the SHIPPING _fill_cusp_tails
  a synthetic EXACT-A3 tail (true_dt=A*theta**(2/3), NaN below theta=0.10),
  confirm fill reconstructs analytic A3 to rtol 1e-9, then fit
  log(s'-s'_cusp) vs log(d) -> slope 0.6667+-0.05. Self-falsification: d^1
  arc-length -> slope~1 (rejected), d^{1/2} -> slope~0.5 (rejected). s'_cusp
  = s'[0]-filled[0] (cumtrapz initial offset). The test exercises the real
  catastrophe-law imposition code, just on a controlled tail.
- F083 NODE-ECONOMY -> PUSHBACK ON THE eps LEG (acceptance-evidence): the
  held-out eps<=0.0237 leg needs a full production tube build (~171s) and is
  UNREACHABLE at any node count anyway (measured eps 0.145@30 nodes,
  0.56@4 nodes on the gamma=0.4/w in {52,60} fixture). Implemented the cheap
  engine-free HALF: cap<48 structural + closed-form derived N at w in {52,60}
  is <48; self-falsification builds count at w=1e5 (raw ceil>>48) and shows
  the clamp pins it to cap<48. FLAG FOR DRIVER: the delay-uniform axis does
  NOT hit eps<=0.0237 at smoke scale -- either the 0.0237 bar assumes a
  larger/eta-narrower band than the fast fixture, or the accuracy claim is
  overstated; needs a post-build driver sweep to confirm, not a unit test.
- STALE-SCHEMA REFUSAL = clean engine-free invariant. _validate_tube_axis_schema
  RETURNS the accepted tag (str), does NOT return None -- self-falsification
  asserts assertEqual(...,_TUBE_AXIS_SCHEMA) not assertIsNone (initial
  assertIsNone FAILED, fixed). _chart_from_npz checks schema BEFORE reading
  the map array, so an old-layout NPZ (key chart0_theta_to_s, tag absent)
  raises ValueError by schema, NOT KeyError -- proves no identity fallback.
  Fixture: _chart_to_npz(chart,0) then json-rewrite chart0_meta (value None
  deletes a key).
- NEIGHBOUR REGRESSION AUDIT (step 7, run + read): my edits touch ONLY
  test_lensing_tube_nyquist_coordinate.py; no regression there. But the WP2
  production rename theta_to_s->theta_to_s_prime HARD-FAILS other-owned
  suites RIGHT NOW in the fast tier -- test_lensing_surrogate.py measured
  7 failed + 2 errored (from_values(theta_to_s=...) TypeError; chart.theta_to_s
  AttributeError; cascading anti-vacuity errors). Cross-suite stranded sites
  (NOT mine to edit, for driver/owners): test_lensing_surrogate.py
  L2694-3233 (NON-gated, failing now); test_lensing_wedge_dd_arclength.py
  L524-528 test_tube_still_exposes_theta_to_s (NON-gated HARD-FAIL, asserts
  'theta_to_s' IN TubeChart fields); test_lensing_surrogate_training.py
  L2652-2929 (gated ShippedArcLengthTubeGridTestCase); test_lensing_log_reach_gamma.py
  L335/364/585/610; test_lensing_low_w_extrapolation.py L194-195. My own
  file's theta_to_s hits (L793/890/894) are the INTENTIONAL stale-key
  fixture (chart0_theta_to_s), not strandings.

## 2026-08-17 build (tube Nyquist coordinate + theta_to_s->theta_to_s_prime rename)
- New suite test_lensing_tube_nyquist_coordinate.py (16 tests, 15s) pins the
  3 Architect specs on surrogate_training._build_tube_chart: (1) DRY delay
  pin — shipping s' == cumtrapz(|grad(Delta_tau)|) reconstruction from
  _airy_fold._merging_fold_pair to 1e-12, Delta_tau=0.5*(tau_minus-tau_plus);
  (2) monotone s' across arc while Delta_tau is non-monotone (interior
  turnover); (3) Nyquist n_theta==ceil(_TUBE_NYQUIST_PPP*w_max*tv/(2pi))
  clamped to [_TUBE_MIN_THETA_NODES=4, n_theta_cap=32], saturates at 32 at
  10x w_max. _engine_envelope is mocked (delay-map runs BEFORE any engine
  call, so all invariant tests are engine-free).
- FIXTURE DERIVATION: full cusp-to-cusp production arc does NOT build —
  _tube_delay_map raises "not strictly increasing" because the servable
  subrange reaches Delta_tau~1e-7 and _fill_cusp_tails clamps near-cusp tail
  to zero (flat s'). Derive a robust sub-arc via dataclasses.replace trimming
  theta_lo/hi to where dt>=8% of peak at both ends. gamma_grid needs >=4
  nodes (TubeChart.from_values validates every axis >=4); drive tv_delay from
  rep_gamma=median(gamma_grid) so builder & oracle stay consistent under
  float-median drift. tiny-w build needs w_lo_eff=min(w_lo,0.5*w_max).
- FLAGGED PRODUCTION DEFECT (for driver/coder): _build_tube_chart is called
  UNGUARDED in _train_band_charts (_load_or_build doesn't catch build
  exceptions), so full-arc tube training crashes on the first arc via the
  same "not strictly increasing" ValueError. Not a test bug.
- WP2 RENAME STRANDED CROSS-SUITE (other-owner scope, NOT edited): non-gated
  test_lensing_wedge_dd_arclength.py::FieldExposureTestCase::
  test_tube_still_exposes_theta_to_s now HARD-FAILS (asserts 'theta_to_s' in
  TubeChart fields; field is now theta_to_s_prime). Gated/rotting-silently:
  test_lensing_surrogate_training.py ShippedArcLengthTubeGridTestCase
  (@_TRAIN_TIER_SKIP, chart.theta_to_s ~L2885/2927/2929);
  test_lensing_surrogate.py (from_values(theta_to_s=...)+chart.theta_to_s
  L2694-3233); test_lensing_log_reach_gamma.py (L335/364/585/610);
  test_lensing_low_w_extrapolation.py (L194-195). Their owners must migrate
  to theta_to_s_prime.
