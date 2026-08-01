# Test Dev Short-Term Observations

- WP1 ((s,d) far-field coord RESTORE) PORT of test_lensing_exterior_admission.py:
  the ONLY breakage was Gamma1BoxCentreGuardTestCase's `_build_guard_chart`
  calling `from_engine(rho_range=,theta_c_range=,n_rho=,n_theta=)` -- retired
  kwargs (new sig: gamma_range,s_range,d_range,w_range,arc_theta_lo/hi,arc_branch,
  n_gamma/n_s/n_d). MEASURED (scratch): the 3 `_build_guard_chart` methods are
  UNPORTABLE-without-changing-assertion (brief acceptance #3), NOT a mechanical
  kwarg swap: (a) new from_engine builds ONE arc-length map over the whole
  gamma grid + runs `_reject_if_cusp_spanning` PER gamma node BEFORE the node
  loop; at gamma=1.0 both RAISE LensDomainError (det A=0 wall) which propagates
  OUT of from_engine -> gamma=1.0 is NOT recorded in refused_points (refusal
  moved to tiler's except in surrogate_training._build_farfield_chart -> ladder
  gap). So no chart-with-refused-gamma=1.0-slab exists. (b) (s,d) is astroid-only:
  _build_farfield_chart REFUSES parity!=1; _reject_if_cusp_spanning(1.3,arc)
  raises (arc outside wedge |sin2th|<=1/gamma); parity==-1 saddle chart has no
  (s,d) analogue. (c) teeth mocked sg._from_caustic_fixed, not on (s,d) path
  (which uses _from_farfield_smooth). RESOLUTION: REMOVED the 3 methods
  (test_box_centre_gamma_one_yields_none_labels /
  test_node_at_gamma_one_is_recorded_refused /
  test_guard_catches_lens_domain_error_specifically) + dead `_build_guard_chart`
  helper + dead GUARD_*/GAMMA1_* consts; KEPT the 2 surviving methods
  (test_caustic_reach_raises_only_exactly_at_one +
  test_guard_is_stable_across_the_boundary_not_a_knife_edge) which exercise the
  STILL-EXISTING scalar `_caustic_reach`/`_to_caustic_fixed`/`_from_caustic_fixed`
  (WP1 did NOT remove them). Rewrote class docstring + added RETIRED note above
  `_rcaustic_table`. NOTE: WP1 kept `_to_caustic_fixed` so the vectorized-vs-
  scalar cross-check (`test_rho_map_vectorisation_matches_production` ~L862,
  `_to_caustic_fixed_vec` vs sg._to_caustic_fixed) is NOT a farfield_smooth
  oracle -> passed, no value finding. Round-trip/Jacobian tests (rho=1 on
  caustic, drho/d|y|=1) ALSO pass (still on the scalar caustic-fixed coord, not
  farfield). File 45->42 tests, 42 passed 3:13, 0 anti-vacuity. Neighbor suites
  (test_lensing_surrogate/ppgo_bandsplit/surrogate_census) STILL use retired
  rho_range from_engine API -> pre-broken by WP1, owned by OTHER runs, NOT
  touched/run (my change is test-only, cannot regress them).

- WP1 (closed-form caustic reach+direction replaces 720-pt scan)
  test_lensing_surrogate.py OWNED suite (served-values + cost claim): appended
  ~530 lines, 6 classes + 3 helpers + WP1_* constants before __main__. KEY
  FINDING: WP1 touches served values ONLY via the macro-saddle path
  (surrogate._caustic_reach -> ppgo_map.caustic_geometry[0]); POSITIVE parity
  uses geometry.r_caustic (WP1-invariant). At SHIPPED saddle gammas
  (1.25/1.35/1.45 in SAD_BOX) the 720-scan was ALREADY converged, so
  served closed-vs-720 AND closed-vs-dense are BYTE-IDENTICAL (max d|F|_rel
  0.00e+00, dphase 8.5e-17). The "moves TOWARD converged" phenomenon lives at
  the REACH level for near-wall gammas 1.001-1.1 (worst 720 rel-err 1.10e-4,
  closed 6.1e5x closer) which the shipped box doesn't cover -> tested reach
  directly, not served. Wp1CausticRadiusOracleTestCase = STAGE-1 oracle
  validation: hand-rolled _wp1_caustic_radius_max (vectorized polar max over
  both sqrt branches, transcribing macro_matrix@x - x/|x|^2) vs shipping
  geometry._caustic_source <1e-13 BEFORE use as dense-scan oracle (F002).
  Wp1ClosedFormReachIsConverged: closed reach==dense 4e5-pt scan to 1e-7;
  surrogate._caustic_reach(g) bit-identical to caustic_geometry(g,0.0)[0].
  Wp1CoarseScanCorrection: where 720 wrong, closed strictly closer (anti-vac
  >=1 wrong gamma; PNG wp1_reach_error_vs_gamma.png).
  Wp1ReachCallCount: monkeypatch COUNTER on geometry.critical_point around one
  served lnlike -> reach path 0 calls, full serve 1 (was 1440); HEAD 720-scan
  witnessed at 1440 via git-show AST-extract of the retired FunctionDef exec'd
  in minimal namespace (SkipTest guard if git/n_theta absent).
  Wp1ServedValuesUnchanged: served tracks converged reach; unchanged vs 720;
  positive parity byte-identical when caustic_geometry poisoned (mock.patch);
  full serve <=10 critical_point calls + ~2.39ms/serve wall-time REPORT (not a
  gate). Wp1SelfFalsification: biased reach fails converged; scanning-stub
  caught by counter (GOTCHA: stub at gamma=1.2 saddle theta=pi/4 is OUTSIDE
  critical wedge -> uncaught LensDomainError + anti-vac tearDown n=0 ERROR;
  FIX = use gamma=0.50 positive parity where all angles valid AND wrap stub's
  critical_point in try/except geometry.LensDomainError: continue); reach error
  moves served above bar; sign-flip transcription slip caught. Added imports
  `import subprocess` + `from cogwheel.lensing import ppgo_map`. AUDIT (WP1
  dropped n_theta from caustic_geometry sig): grepped ALL of this file - every
  n_theta= hit is unrelated surrogate-TRAINING grid (from_engine/_train/
  _accuracy_tube_chart lines 456/518/896/2497/2902/3047/3063), NO existing test
  called caustic_geometry(...,n_theta=...) -> signature change breaks nothing
  here. 16 new tests; FULL FILE 78 passed/1 skipped 2:37 (under 5min ceiling),
  no regression.


- WP1 (closed-form caustic_geometry replaces 720-pt scan) exterior_admission
  suite: added 2 classes to test_lensing_exterior_admission.py.
  Wp1ClosedFormParityWallTestCase (6 tests) pins the NEW closed form DIRECTLY
  on ppgo_map.caustic_geometry (not just the sg._caustic_reach wrapper):
  parity wall |gamma|==lam is exact-point refusal (cg(1.0,0.0) raises; both
  nextafter(1.0,+/-) finite reach + unit direction); wall TRACKS lam not
  hardcoded 1 (kappa=0.5 -> wall at gamma=0.5, gamma=1.0 now served);
  over-critical lam=1-kappa<=0 (kappa>=1: 1.0/1.5/2.0) refuses, kappa one-ULP
  under 1 served; reach DIVERGES monotone toward wall both sides (offsets
  0.1/0.01/0.001/0.0001 -> below 5.69/19.8/63/200, above 2.08/6.99/22.3/70.7;
  near-wall floor 50, far ceiling 10 => finite-divergent-finite); scalar
  wrapper sg._caustic_reach(g)==cg(g,0.0)[0] assertEqual bit-identical.
  Diagnostic PNG wp1_caustic_reach_hole_at_gamma_one.png (semilogy, hole at
  1.0, no gamma==1.0 sampled). Self-falsif class patches ppgo_map.
  caustic_geometry with never-raising stub -> wall-refusal check raises
  AssertionError (teeth); positive control real form passes. KEPT existing
  Gamma1BoxCentreGuardTestCase (single-point contract ~line 1058, admission
  labels-None + node-refusal-at-gamma=1) UNTOUCHED - it already certifies the
  admission-decisions-unchanged half; my class is the cheap closed-form half
  (no engine builds). Added top import `from cogwheel.lensing import ppgo_map`.
  New 7 tests 3.9s; full file 45 passed 3:14 (under 5min ceiling). AUDIT: WP1
  kept the (reach,direction) tuple sig + LensDomainError refusals; grepped ALL
  tests/ - every caller uses cg(gamma,kappa)[0] or reach,direction=cg(...);
  ghost/ppgo_map/bandsplit/exterior_windows/surrogate* suites (other-run-owned)
  consume the unchanged API -> no signature breakage, nothing to fix.
  Test-only change cannot regress production; neighbors not re-run (heavy,
  unchanged API).

- WP1 TubeChart arc-length: added 4 classes to test_lensing_surrogate.py
  (ArcLengthMapRoundTripTestCase, ChartSplinesInArcLengthTestCase,
  TubeChartMapSerializationTestCase, ArcLengthSelfFalsificationTestCase).
  Build real arcs via surrogate_training._astroid_arcs/_saddle_arcs +
  _tube_arc_length_map(rep_gamma, arc, n_map=2001). Saddle neg-wedge arc =
  _saddle_arcs(1.30)[arc0] measured [-0.352,-0.132] (inside [-0.39,-0.09]).
  Round-trip err ~2-3e-16 (gate 1e-6). GOTCHA: the s(theta(s)) np.interp
  round-trip is ~0 for ANY monotone table (self-consistent even if you
  reverse a chunk) — its teeth come from the strictly-increasing/endpoint
  assertions, NOT the numeric bound; make the bound reachable-red via a
  MISMATCHED-row round trip (forward uses s*1.05, inverse uses s) -> 0.05.
  Spec C: served==_contract at v2=interp(theta,map) bit-identical (0.0),
  differs from v2=theta_inframe by rel ~0.54 (use non-affine speed
  2+1.5sin so s spans [0,2] while theta spans [0.2,1.2], both inside s
  knots -> no extrapolation). Perturbed-map served delta ~0.088.
  Full file 58 passed/1 skip in 3:45. Sibling TubeChart.from_values calls
  (census, farfield_envelope) use identity-map default -> unbroken by WP1.
  Needed new top import: from scipy.integrate import cumulative_trapezoid.

- WP1 shard 2: added 2 classes to test_lensing_surrogate.py.
  CoordinateChangeAccuracyTestCase: fit chart to analytic surface
  _analytic_smooth_in_s(w,gamma,u,s) (closed form, F002-independent) at
  ACCURACY_N_THETA=14 nodes via _accuracy_tube_chart + _nonaffine_map; serve
  on cusp-free theta sweep vs target at s=interp(theta,map). MEASURED arc rel
  ~2.0e-4 (converged n_theta 12->16 unchanged), gate 0.05 (F016 complex).
  Positive control = contract SAME chart at raw theta -> rel ~0.54 (>0.20
  ACCURACY_RAW_THETA_MIN) = reachable red proving coord choice load-bearing.
  IdentityDefaultBackCompatTestCase: _identity_default_tube_chart() =
  from_values with NO s_grid/theta_to_s (legacy form) -> identity map branch
  s=theta-theta_lo. Pinned map row0==theta_grid, row1==theta_grid-theta_grid[0]
  (both bit-equal). Served F frozen as float.hex golden literals IDENTITY_GOLDEN
  (3 queries x w-idx {0,4,9}); compare via float.fromhex assertEqual bit-exact
  (NO git HEAD, NO oracle). Self-falsification test: 0.1%-perturbed literal
  fails. New 2 classes = 5 tests, all green. Full file 62 passed/1 skip 3:47.
  AUDIT: from_values map-less callers (census, farfield_envelope siblings +
  this file's pos_tube/sad_tube recon fixtures) UNBROKEN by identity default;
  no bare TubeChart(...) ctor calls in tests so required theta_to_s field
  breaks nothing.

- WP1 shard 3 (adequacy diagnostic) surrogate_TRAINING suite: added
  SingleGammaMapAdequacyTestCase (FAST engine-free, NOT _TRAIN_TIER gated, 5
  tests, 4.3s) to test_lensing_surrogate_training.py. Spec: single-gamma
  arc-length map adequacy. Helper _wp1_normalized_arclength(g,arc) wraps
  training._tube_arc_length_map -> (theta_fine, s_fine/s_fine[-1]); theta_fine
  is np.linspace(theta_lo,theta_hi,n_map) GAMMA-INDEPENDENT (only s_fine varies)
  so both edges compare point-by-point, NO interp. F042 band=(1.52,1.58)
  n_gamma=4 -> gamma_grid[0]=1.52, [-1]=1.58; MEASURED edge gap=0.0054 < bar
  _WP1_ADEQUACY_BAR=0.05; midpoint(1.55) vs each edge ~0.0026/0.0028.
  SELF-FALSIF/positive control _WP1_WIDE_PARITYWALL_BAND=(1.05,1.9) on SAME
  F042 arc: gap~0.093 > 0.05 (teeth). GOTCHA: wedge |sin2theta|<=1/gamma closes
  at gamma~1.976 for this arc's theta range [-0.265,-0.101]; gamma>=1.98 raises
  LensDomainError, so upper control edge must stay <1.976 (1.9 works, 2.0 dies).
  s_hat[0]==0.0 exact, s_hat[-1]==1.0 (x/x), strictly increasing pinned.
  Diagnostic PNG tests/output/wp1_single_gamma_adequacy.png (2-panel edges vs
  wide). No name collision with prior _WP1_* / _wp1_*. Test-only additive edit;
  full file fast tier 31 passed/48 skipped 6.2s. No production edits -> sibling
  test_lensing_surrogate.py (other-run-owned) cannot regress; not run (heavy).

- WP1 (build-path side) surrogate_TRAINING suite: added 4 classes to
  test_lensing_surrogate_training.py (arc-length node-PLACEMENT half; the
  schema/serialization half lives in sibling test_lensing_surrogate.py).
  ArcLengthNodePlacementGeometryTestCase (FAST engine-free, 7 tests):
  reproduces _build_tube_chart placement via training._tube_arc_length_map +
  np.interp; nodes uniform-in-s under BOTH production map (rel~1e-16) AND an
  INDEPENDENT F002 polyline oracle _wp1_polyline_arclength (Euclidean segment
  sum of geometry._caustic_source vs code's caustic_speed+cumulative_trapezoid;
  agree ~3e-8). Self-falsif: uniform-THETA grid is ~0.14 non-uniform in true
  arc length (>0.05) -> teeth. Endpoints assertEqual arc.theta_lo/hi.
  3 engine classes @_TRAIN_TIER_SKIP (COGWHEEL_TRAIN_TIER=1, 15 tests, 3:39):
  ShippedArcLengthTubeGridTestCase reuses _wp3_fixture()['on'], rep_gamma=
  median(gamma_grid)=1.55, checks chart.theta_to_s carried map.
  ArcLengthNodeEfficiencyTestCase: arc eps(n=4)=0.0279 < frozen 0.059 literal
  (_INCUMBENT_EPS_N4, NO git-show), arc<uniform every n (4/5/6), arc clears
  0.05 bar at n=4 vs uniform n=5.
  SPEC-2 PREMISE FALSIFIED (F042): spec claimed arc-length eps swing<0.05 under
  +-0.01rad bound shifts vs uniform ~0.20; MEASURED arc swing=0.31 > uniform
  0.25 -> did NOT assert the false claim. Encoded the REAL knife-edge-gone =
  registration-bar MARGIN: arc <0.05 every variant (max 0.036), uniform TRIPS
  it (max 0.073); test_spec2_swing_premise_is_false locks measured reality.
  NAMING: file already had unrelated _WP1_* (winding/far-field from a prior
  build's WP1); my _wp1_/_WP1_ARCLEN* have distinct suffixes, no collision
  (import verified). AUDIT: every existing TubeChart.from_values is MAP-LESS ->
  identity-default branch -> unbroken by new optional theta_to_s; no bare
  TubeChart(...) positional ctor anywhere; no pre-WP1 test pins _build_tube_chart
  to uniform-theta. Ran gated SaddleTubeTailTestCase (shares _build_tube_chart)
  under TRAIN_TIER: 5/5 green -> arc-length gain did NOT rescue fix-off chart
  below bar (off_eps>0.05 holds). 4 diagnostic PNGs tests/output/wp1_arclen_*.
  Fast tier 7 passed/15 skipped 4.4s.

- WP1 (closed-form caustic_geometry reach+direction) CROSS-SUITE re-pointing
  audit for test_lensing_ghost.py (MY owned suite). RESULT: NO literal moved,
  ZERO edits. Only consumer of caustic_geometry in ghost suite is _anchor_source
  (~L573): `reach,direction=ppgo_map.caustic_geometry(gamma,kappa)` -> rho*reach
  along direction rotated by angle; feeds ChangRefsdalChannels physics. Grepped
  whole file: NO direction-quadrant or reach VALUE is asserted anywhere; anchor
  gates are tolerance-based (||C|/|E|-1|<0.10, |arg(E/C)|<3.5deg) with measured
  (0.110,0.111,1.5deg) only in COMMENTS. Ghost suite never passed n_theta so the
  WP1 SIGNATURE change (dropped n_theta) breaks nothing. WP1 makes direction
  EXACT on-axis for positive-parity folds (gamma 0.2/0.4, e<1 -> winning cand is
  axis cusp u=1+-e -> cos2theta=+-1 -> theta 0 or pi/2 -> [axis_a,0] or [0,axis_b],
  canonicalized) vs old 720-pt scan's ~0.5deg-off direction; this only STRENGTHENS
  the existing test_exactly_on_axis_is_refused (angle=0 now truly on principal axis
  -> diagonal matrix -> removable singularity u=a22 -> GhostDomainError). Ran green
  under landed WP1: ghost 31 passed/1 xfailed 4.7s. Neighbor WP1 consumers (also
  green, owned by other runs, left untouched): ppgo_bandsplit 62 passed/4 skip
  25s; exterior_windows 75 passed/1 xfailed 2:47. So NO literal moved in any of
  the 3 named suites -> all left untouched per spec.

- WP1 specs 4-5 (sanity literal + off-axis direction) OWNED suite
  test_lensing_ppgo_map.py shard 2: appended 3 classes + 2 helpers + 7
  constants (7 new tests 4.7s; full file NOW 37 passed 9.5s). SPEC4
  CausticReachSanityLiteralTestCase: reach==2*g/sqrt(1-g) (INDEPENDENT algebra
  reduction of u=1-e axis cusp: numer=4g^2(1-g), u^2=(1-g)^2 -> 4g^2/(1-g);
  MEASURED rel ~1e-16 <=1e-9) over g=0.3/0.6/0.9; direction=[0,1] EXACT (axis
  aligned, min(|d0|,|d1|)=0.0). GOTCHA: SPEC.md literal 5.692100 is TRUNCATED,
  differs from exact 5.692099788303083 by 2.117e-7 which EXCEEDS 1e-9 -> tight
  1e-9 gate vs EXACT closed form, loose 1e-6 straddle vs SPEC literal.
  Quadrant NOT pinned. SPEC5 OffAxisDirectionAgreementTestCase: g in (1.05,1.1)
  diagonal saddle band; _wp1_scan_direction (argmax of _wp1_parametric_radius
  over both branches, direction via geometry._caustic_source - INDEPENDENT
  path) vs closed dir, reduced mod 4-fold via _wp1_axis_reduced_angle (abs-fold
  to Q1 then acos). MEASURED axis-reduced angle=0.000e+00 (dot clips>=1;
  maximiser dir is smooth fn of g so converged scan pins it <<grid step) - bar
  2pi/11520=5.45e-4 generous, canary worst<1e-3. Genuinely-diagonal guard
  min-comp>0.05 (0.138@1.05,0.285@1.1). Wp1DirectionSelfFalsificationTestCase:
  rotate dir 0.1rad -> reduced angle==0.1 (>>bar), 0.1%-wrong reach>1e-9,
  diagonal [.707,.707] min-comp 0.707>1e-9. PNG wp1_offaxis_direction_overlay.
  AUDIT: WP1 kept caustic_geometry(gamma,kappa)->(reach,dir) sig + values
  numerically unchanged (dropped internal n_theta never exposed); prior shard
  already grepped ALL tests/ - neighbors recompute reach, no literal to shift.
  Only edited my owned file, no prod edits.
- WP1 (closed-form caustic_geometry reach+direction) OWNED suite shard 1
  test_lensing_ppgo_map.py: appended 4 classes + 3 module helpers (8 new
  tests, 4.5s; full file 30 passed 9.3s). Grid _WP1_GRID = 10 (gamma,kappa)
  spanning both parities incl near-wall saddle 1.001..1.2 + cusp switch
  1.177651 (1.05 offaxis, 1.3 onaxis) + kappa=0.2 cases.
  SPEC1 ClosedFormReachVsParametricScanTestCase: INDEPENDENT F026 |y|(theta)
  parametric caustic-radius scan _wp1_scan_reach (NOT a source-plane ring -
  a ring misses the thin near-wall spike). Two-stage oracle: stage-1
  test_parametric_radius_matches_production_caustic_source validates my
  hand-rolled radius vs geometry._caustic_source to 1e-12 BEFORE using it.
  GOTCHA: n_theta=11520 (Prof floor) gives rel 3.1e-7 at gamma=1.001 (>1e-7
  bar) -> use _WP1_SCAN_N_THETA=46080 (pure brute scan, no refine): MEASURED
  worst rel=1.54e-8. Bar _WP1_REACH_RTOL=1e-7; also assert worst<5e-8.
  SPEC2 ReachMaximiserStationarityTestCase: recover winning theta MACHINE-
  PRECISELY from analytic winning-u (NOT radius-max scan, which only gave
  ~1e-8 theta -> ratio ~1e-7) via cos2theta=(u^2-1+e^2)/(2eu), branch=sign(
  u-e*cos2theta). Stationarity ratio=|2 y.y'|/|y|^2 with y=_caustic_source,
  y'=caustic_derivatives (independent code path). DISJUNCTION bar: ratio<=
  _WP1_STATIONARITY_RATIO_BAR=1e-9 OR speed<_WP1_CUSP_SPEED_FLOOR=1e-4.
  MEASURED 9 ratio-arm (~1e-13..0.0) + 1 floor-arm (gamma=0.9 axis cusp:
  ratio 4.47e-8, speed 1.27e-7 numerical-zero) -> both arms load-bearing,
  asserted via assertGreater on both counts. Wedge u=sqrt(e^2-1) EXCLUDED
  (F044 y' diverges); assertNotEqual(label, wedge) confirms it's never the
  winner. tie-back test: recovered reach==caustic_geometry reach (places=12)
  + direction PARALLELISM |recovered.direction|==1 (NOT exact-component -
  axis cusps get ~1e-33 numerical x-component flipping canonical sign).
  SPEC3 SingleSourceReachEqualityTestCase: _scalar_caustic_reach(gamma)==
  caustic_geometry(gamma,0.0)[0] assertEqual bit-exact over 6 gammas both
  parities. SELF-FALSIF Wp1SelfFalsificationTestCase (3 teeth): coarse n=360
  scan MISSES near-wall spike at 1.001 (rel>1e-7) while 46080 passes; theta
  offset +0.12rad breaks BOTH stationarity arms; nextafter breaks bit-equal.
  2 PNGs tests/output/wp1_reach_closed_form_vs_scan.png (coarse-ring-misses-
  spike diag) + wp1_stationarity_ratio_vs_gamma.png. AUDIT (READING): WP1
  kept caustic_geometry(gamma,kappa)->(reach,dir) sig + values numerically
  unchanged; caustic_derivatives/_caustic_source/_caustic_reach untouched.
  Grepped ALL tests/ - every neighbor RECOMPUTES reach (no hardcoded reach
  literal to shift); tightest gate exterior_windows:783 places=12 is SELF-
  referential (both sides = same rewritten fn). Ran ghost 31p/1xf 4.2s +
  exterior_windows reach/caustic subset 14p 12s -> green. Nothing to fix.

- Current WP1 far-field compatibility port (`test_lensing_surrogate.py`, `test_lensing_surrogate_lobe.py`): migrated the synthetic multi-chart fixture from retired `(rho, theta_c)` tensors to physical-box-derived `(s, d)` axes plus a gamma-resolved `_FarFieldArcMap`; mapped the stored physical refusal through the same serve map; changed `select_chart` / `_farfield_serves` calls to pass eigenframe source coordinates only. `ChartSelectionTestCase` 5 passed and `LobeExclusivityTestCase` 2 passed under `cogwheel-newlal` (only h5py HDF5 warning). The lobe golden chart’s old axis artifact/digest literals remain deliberately untouched pending Professor direction: current `(s,d,arc_map)` serialization necessarily changes that frozen artifact, so do not regenerate/weaken literals without approval.

- Professor-directed lobe golden rebaseline (current `(s,d)` FarFieldChart): retained the incumbent analytic field as `_positive_physical_envelope(logw, gamma, y1_eig, y2_eig)`, deriving retired rho/theta from every physical point; sampled it at `(_POS_GAMMA_GRID, _POS_S_GRID, _POS_D_GRID)` inverse-map nodes on a gamma-resolved astroid map. Fixed source is off-grid exterior `(s,d)=(0.422949...,0.15)`, physical spline residual 5.46e-3 under 2e-2 bar. New frozen served hex and content digest `d5c81aaf...d80e388`; 11 focused lobe checks (golden + lobe exclusivity) passed in 18.50s. Added current schema/strict map/non-identical gamma rows, save/load bit fidelity, and current-schema node/map perturbation teeth tests.

- WP1 lobe V1 identity-path byte-identity test (test_lensing_surrogate_lobe.py):
  added LobeV1IdentityPathTestCase (7 tests, 4.1s). Builds SYNTHETIC V1 lobe
  chart via from_lobe_values(theta_to_s=None, s_grid=None) with genuine
  admission geometry + random envelope data (seed 20250801). Tests:
  (1) chart.theta_to_s is None, (2) _evaluate_chart returns finite (no crash
  from None indexing), (3) save/reload yields theta_to_s=None, (4) saved meta
  carries _LOBE_AXIS_SCHEMA_V1 tag, (5) no chart0_theta_to_s key in npz,
  (6) served values BYTE-IDENTICAL before/after save-load (core identity
  guarantee), (7) mutation: indexing None raises TypeError (proving inverted
  guard would crash). Full file 54 passed 50s. BACKWARD AUDIT: existing
  sqrtedge tests (LobePersistenceTestCase, LobeSqrtEdge*) all use engine
  charts WITH theta_to_s -> unaffected. LobeSqrtEdgeBoundShiftMarginTestCase
  already exercises from_lobe_values(theta_to_s=None) for UNIFORM charts ->
  confirms V1 path was already working. No breakage found.

- Current far-field `(s,d)` test port: macro-saddle `FarFieldChart` is forbidden by DATA_CONTRACTS; synthetic multi-chart fixture now has positive Tube/FarField + saddle Tube only, and saddle far-field query asserts exact fallthrough. Domain refusal seam and serialization use chart `(gamma,s,d)` + gamma-resolved arc_map, no raw rho/theta. Professor beta oracle: compare Etilde(beta) bitwise to Etilde(q_hat, beta=0) for the actual floating rotated eigenframe source; separately bound q_hat-q_nominal from float64 rotation/orthogonality, never widen legacy 1e-12 envelope delta. Physical off-grid witness has F-normalized label error ~7.08 under the narrow one-foot chart, so the 1e-3 held-out assertion is held for Professor/production direction; do not lower it.


- Professor-certified synthetic-chart separation: in `test_lensing_surrogate.py`, removed the obsolete `POS_RECON_TOL=0.2`/saddle `0.05` and the synthetic chart's held-out/refinement assertions (measured off-grid label eps ~7.083, not certifiable). The local `(s,d,arc_map)` fixture now explicitly covers only node-exact beta covariance, a fresh-engine node-label round trip (F-normalized <=2e-9; measured max 1.21e-9 across beta), domain/refusal, serialization, and macro-saddle exact fallthrough. Certified 1e-3 off-grid positive exterior held-out accuracy, node convergence/refinement, and real-coefficient corruption falsifier stay in `test_lensing_farfield_envelope.py` (`EXTERIOR_TILE_CENTER=(1.5,1.5) +/- .2`, gamma .02-.06). Focused node-label test passed under `cogwheel-newlal`; `py_compile` and collect-only passed (71 tests).

- WP1 lobe s-coordinate (sqrt-edge) test_lensing_surrogate_lobe.py: added 3 new
  classes + 2 helper functions + 3 module constants. LobeSqrtEdgeCoordinateRoundTripTestCase
  (5 tests): verifies theta_to_s[1,0]==0.0 exact, matches closed-form oracle
  (s=sqrt(span)-sqrt(theta_max-theta)) to 0.0, round-trip err 0.0 (at 2001 nodes
  interp is self-consistent), strict monotonicity, shape (2,2001).
  LobeSqrtEdgeBoundShiftMarginTestCase (3 tests): at the smoke fixture (7 nodes,
  theta span ~0.37rad) BOTH coords give ~0.138 eps (tile not at cusp, so no
  knife-edge manifests). Encoded: sqrtedge swing < 0.01 across ±0.01rad shifts
  (measured 0.003), shifted-chart oracle matches closed form, sqrtedge ≈ uniform
  (no degradation). The 0.05 bar IS a production-scale phenomenon requiring 12+
  nodes at a cusp-adjacent tile — unreproducible at smoke scale.
  LobeSqrtEdgeSelfFalsificationTestCase (2 tests): wrong-orientation formula
  (s=sqrt(theta-theta_min)) differs from production; mismatched forward/inverse
  map exceeds round-trip bar. ALSO: added theta_to_s persistence assertions to
  existing LobePersistenceTestCase (bit-for-bit + not None + sqrtedge schema in
  npz meta). BACKWARD COMPAT FIX: widened _NODE_EXACT_TOL from 1e-10 to 1e-7
  because WP1 spline is on s-coordinate and serve does theta→s interp (~6e-9
  error at theta nodes). 47 passed 51s.