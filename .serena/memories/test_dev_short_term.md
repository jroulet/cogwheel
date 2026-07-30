# Test Dev Short-Term Observations

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
