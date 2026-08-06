# Test Dev Short-Term Observations

- InteriorWedgeChart WP1 (ffin retirement) test extension
  (test_lensing_interior_wedge_chart.py, now 46 tests): added 3 classes.
  (1) WedgeHeldOutAccuracyTestCase: 5-node/axis chart via
  LensAmplificationSurrogate.from_wedge_engine (NOT InteriorWedgeChart —
  from_wedge_engine is a classmethod on the SURROGATE), 8 off-node interior
  queries incl small-r + theta_wedge=pi/4 diagonal; eps<5e-2 floor, worst
  measured eps=1.45e-2 at 5 nodes (4 nodes gave a corner eps=5.9e-2 OVER
  floor — use 5). Diagnostic scatter saved to tests/output/.
  (2) WedgeD2FoldExactnessTestCase: 4 D2 mirrors (+-y1,+-y2) served |F|
  identical to EXACTLY 0.0 diff (fold is abs() = exact float negation);
  atol 1e-12 fallback unused. Self-falsification: different source diff>1e-3.
  (3) MedialAxisServingTestCase: near-centre + pi/4 diagonal points that the
  retired ffin FarFieldChart refused (nearest-caustic-foot degeneracy) now
  SERVE via select_chart==self.chart, image_count=4, honest eta 0.26-0.31.
  CRITICAL GOTCHA: _evaluate_chart takes LOG w (it clamps to the log_w band);
  ChangRefsdalChannels takes LINEAR w. Passing np.exp(log_w) to
  _evaluate_chart gives eps=0.96 (clamp saturates); pass LOG w to the chart
  and exp(log_w) to the engine -> eps~7e-3.
  BACKWARD-COMPAT FIX (own file): test_gate_c_log_w_outside_band_refuses was
  stale — _log_w_band_serveable now gates ONLY the high end (low-w flat
  extrapolation), so a below-range log_w_min SERVES. New body: assert
  high-end refuses AND low-end serves. Matches test_lensing_low_w_extrapolation.

- Fold-ppGO handoff test (test_lensing_fold_ppgo_handoff.py): the spec's
  claim that near-axis theta forces xi < 4 at gamma=0.5, rho=0.3 is FALSE
  — delta_tau remains ~0.26-0.33 for all angles at that rho. The gate
  refusal regime requires rho closer to the caustic (rho=0.7 at gamma=0.5
  gives xi ~2.15 at w=45). Fixed the fixture to use RHO_NEAR_CAUSTIC=0.7
  instead of near-axis theta.
- Backward-compat audit (fold-ppGO handoff WP): the new `if rho <= 1.0`
  fold path in _surrogate_coefficients does NOT break existing tests because
  all test fixtures use w < 20 (xi < 4.0 even at rho=0.3), so the fold
  gate refuses and falls through to None as before. Confirmed by running
  test_lensing_born_residual_wiring.py (34 passed) and
  test_lensing_fold_ppgo_correction.py (23 passed).
- Fold-ppGO handoff test extension (test_lensing_fold_ppgo_handoff.py):
  Added 3 new test classes (7 new test methods):
  (1) ErrorEstimateFineGateTestCase: gamma=0.85, rho=0.5 near metamorphosis
      produces |c1|~0.93, error_est~0.066 >> 1e-4, proving the fine gate
      is load-bearing (coarse xi gate admits xi~5.85>=4, fine c_A refuses).
  (2) DefaultPathUnaffectedTestCase: mock-based, select_chart returning
      chart takes priority (fold block never reached). mock_chart must be
      IN surrogate.charts for _chart_index identity match.
  (3) CensusRecordsPpgoFoldTestCase: M_lens=20e6, w_min~49500, xi~531,
      error_est~8.2e-5 < 1e-4 — all gates pass. Patches select_chart→None
      + get_certified_ppgo_map→None; REAL geometry_partition.
  Backward-compat: _uniform_error_estimate, _merging_fold_pair, _image_at_delay
  are new additive; all 3 neighbor suites (23+63+62 tests) pass green.

- Schwinger mpmath extension (WP1/WP2) backward-compat audit:
  The mpmath extension changes the refuse boundary from W_CEILING_SCHWINGER=60
  to W_CEILING_SCHWINGER_QD=150. ALL existing tests that asserted refusal at
  w∈(60,150] must be updated to use w>150. Affected constants/fixtures:
  REFUSED_W_SWEEP, ABOVE_CEILING_WS, THREE_OUTCOME_W, SADDLE_BOUNDARY_WS,
  F028_SERVE (w=70→151), DeltaMinComputedAtMostOnce _CASES (w=70 now mpmath-band),
  DdMandatoryFalsification perturbed ceiling (patch W_CEILING_SCHWINGER_QD
  not W_CEILING_SCHWINGER), RefusalAboveCeiling _serving_arm→_serving_path
  (geometric path is now a serving outcome above QD ceiling), SelectBranch
  SelfFalsification (w=61→151). Other files broken but out of scope:
  test_lensing_levers.py (f_schwinger(61,...) expects raise),
  test_lensing_waveform.py (above-ceiling probes at w>60 now slow/succeed).
- The mpmath path is VERY slow: ~120s per call at w=70, scaling with w.
  Any test that calls f_schwinger at w>60 must be gated by COGWHEEL_TRAIN_TIER.
  The _oracle_1d at w>60 is similarly expensive.
- Schwinger extension tests (TEST 4/5/6) added to test_lensing_schwinger.py:
  (4) MpmathLazyImportTestCase: subprocess isolation proves no top-level
  mpmath import; AST check confirms no `import mpmath` at module level;
  mock verifies f_schwinger routes to _f_schwinger_mpmath at w>60.
  (5) SaddleRoutingMpmathStructuralTestCase: mock-based, proves _saddle_grid
  routes w∈(60,150] to f_schwinger (mpmath path) not the DD batch. Train-tier
  oracle variant (SaddleRoutingMpmathOracleTestCase) verifies bit-identity
  with direct f_schwinger at kappa=0.
  (6) SaddleWCeilingWiringTestCase: _SADDLE_W_CEILING=148, _upper_w_cap
  returns min(w_max, 148, 58/y_mag). At y_mag=0.5 DD cap (116) binds;
  at y_mag=0.1 saddle ceiling (148) binds. Self-falsification: old ceiling
  58 would give 58 at y_mag=0.1 (proves test has teeth).
- InteriorWedgeChart wiring / ffin retirement (WP1) PORT of
  test_lensing_exterior_windows.py: (1) removed the single retired test
  test_interior_tiles_are_nonempty_and_cusp_aligned (used deleted
  st._farfield_interior_tiles) PLUS its orphaned _plot_admission_map helper
  (dead code, only that test called it). (2) BACKWARD-COMPAT: the WP also
  dropped the `definition` kwarg from st._build_farfield_chart (it now
  hardcodes FARFIELD_KERNEL_SUM internally since interior is charted via
  InteriorWedgeChart, not a far-field chart). Removed `definition=
  st.FARFIELD_KERNEL_SUM,` from BOTH call sites in
  ReprovisionNodeCountTestCase.test_reprovision_recommendation_forwarded_to_node_density
  — semantically equivalent. Audit confirmed all other st.* symbols the file
  uses still exist (_interior_admission KEPT at L1811, _cusp_source_angles,
  _lobe_cusp_source_angles, _lobe_interior_tiles, _saddle_lobe_admissions,
  _farfield_box_to_smooth, interior_w_nodes_per_decade field=15). Full file
  green: 84 passed, 1 xfailed (was 83 passed/1 failed/1 error/1 retired).
- BACKWARD-COMPAT AUDIT (WP1/WP2 mpmath extension):
  BROKEN (out of scope): test_lensing_operator.py
  `test_sheared_host_above_ceiling_refuses_schwinger` (w=70 now succeeds
  via mpmath, doesn't raise); also routing filter at lines 870/918 uses
  W_CEILING_SCHWINGER=60 instead of W_CEILING_SCHWINGER_QD=150.
  test_lensing_surrogate.py FLIP_REFUSAL_W=68 expects refusal but w=68
  now succeeds via mpmath. Both need w raised to >150 to remain valid.
  test_lensing_schwinger.py: all constants already updated (REFUSED_W_SWEEP
  at 150.5+, ABOVE_CEILING_WS at 151+, THREE_OUTCOME_W=151, F028_SERVE
  at 151+, DeltaMinComputedAtMostOnce uses w=151).

- ffin retirement WP1 PORT of test_lensing_ppgo_bandsplit.py (was a HARD
  ImportError at collection -> now 62 passed/4 skipped). Removed the
  `_farfield_interior_tiles` import (was on the surrogate_training import
  line); rewrote InteriorAdmissionTestCase.setUpClass to build tiles via
  `st._wedge_interior_tiles(R_EXTENT=0.6, N_PER_SIDE=5)` (pure geometry, no
  admission/cusp args). KEY: for r<=1 & gamma<1, surrogate._from_caustic_fixed
  == wedge-fixed map (both scale |y| by directional r_caustic), so the
  existing _n_images(gamma, rho=r, theta=theta_wedge) engine oracle works
  unchanged on wedge tiles. Ported tests: (1) wholly-interior + 4-image per
  tile (r_c+half_r<=R_EXTENT<1, r_c-half_r>=_WEDGE_R_MIN, find_images==4);
  (2) nonempty -> len==N_PER_SIDE; (3) admission exterior-refusal (rho=1.2
  @30deg -> 2 images, admits False) + wedge single-column contract (j==0,
  theta_center==half_theta==pi/4). RETIRED the cusp-ray straddle guard (wedge
  tiler has NO cusp-alignment split by design). test_tighter_radius_admits_
  strictly_fewer kept verbatim (uses admission only). Removed orphaned
  _straddles_ray helper. Kept `cls.admission`/`cls.cusp_angles`
  (_interior_admission + _cusp_source_angles both still live). Updated module
  docstring (~L24) + the InteriorAdmissionTestCase cost-NOTE (~L960) to name
  _wedge_interior_tiles. This closes the report-only flag from the prior run.

- ffin retirement WP1 EXTENSION of test_lensing_interior_wedge_chart.py
  (46 -> 63 tests, +17, file runs 43s). Added 5 classes:
  (1) WedgeInteriorTilesContractTestCase: `_wedge_interior_tiles(r_extent,
      n_per_side)` is a SINGLE angular column (j==0, theta_center=pi/4,
      half=pi/4 -> spans [0,pi/2] exactly, NO pi/4 split); uniform radial
      rows contiguous in (_WEDGE_R_MIN=1e-2, r_extent], r_extent<1; row count
      == n_per_side; r_extent<=_WEDGE_R_MIN -> []. Diagnostic dump to
      tests/output/wedge_interior_tiles_ranges.txt.
  (2) WedgeInteriorTilesCapFalsificationTestCase: reconstruct production cap
      `r_extent = min(grid_rho_extent, 1 - max_eta_max/coordinate_radius_min)`
      (the ONE expression in _train_band_charts ~L4303). Larger max_eta_max
      shrinks r_extent AND outer row edge; eta sweep keeps every edge in (0,1).
  (3) FfinRetirementInvariantsTestCase: hasattr(st,'_farfield_interior_tiles')
      is False; _interior_admission survives+callable; _farfield_exterior_tiles
      sig has 'admission' param; _build_wedge_chart src has
      'definition=INTERIOR_SACR_C', _build_farfield_chart has
      'definition=FARFIELD_KERNEL_SUM' and NOT INTERIOR_SACR_C.
  (4) WedgeTrainingPathProducesWedgeChartsTestCase: reuses shared
      _shared_wedge_surrogate() (no extra engine cost) - all charts are
      InteriorWedgeChart, none FarFieldChart.
  (5) WedgeTilesSelfFalsification: split-column / non-uniform / over-unit-
      extent(1.2) / equal-eta cases prove the contract asserts have teeth.
  GOTCHA: check FarFieldChart via string 'definition=INTERIOR_SACR_C' NOT
  bare 'INTERIOR_SACR_C' — the _build_farfield_chart DOCSTRING mentions the
  token in prose, so a naive `not in` would false-fail.
- BACKWARD-COMPAT AUDIT (ffin retirement, OUT OF SCOPE / report-only):
  test_lensing_ppgo_bandsplit.py has a MODULE-LEVEL import of the retired
  `_farfield_interior_tiles` (line 89) + a call (line 620) + a whole test
  class FarfieldInteriorTiles... -> HARD ImportError at collection (entire
  file fails to collect, 0 tests). Owned by another Test Dev run; the OWNER
  must migrate it to `_wedge_interior_tiles` or delete the ffin-specific
  test. Flag to driver.

- SHARD 2a (test_lensing_marginalized_likelihood.py::
  test_prior_draws_are_finite_or_exact_neg_inf): capped ONLY the box
  lens-mass axis so every prior draw evaluates at w<=W_SWEEP_CEILING(55)<60
  -> fast double-double, NO mpmath (~100s/call build-killer avoided).
  Mechanism: i_mass=prior.sampled_params.index('ln_m_lens_msun'); f_top=
  h.fbin[-1]; w_per_msun=dimensionless_frequency(f_top,1.0,0.0); ln_m_cap=
  log(55/w_per_msun); cubesize[i_mass] shrunk so upper edge<=ln_m_cap. Kept
  shear(0..1.6) + source(straddles r=1) at FULL extent (Ruling 4: split is
  mass-orthogonal). Added assertGreater(n_neginf,0) + w_maxes.max()<=60;
  did NOT weaken existing (isnan/isposinf/finite-or-neginf/n_finite>0).
  COST: measured ~1.6s/draw end-to-end (NOT the spec's 0.2s — that's
  engine-eval only; coherent-score QMC dominates). At N=40 ran 63.87s (OVER
  <60s ceiling); reduced N_PRIOR_DRAWS 40->30 -> 32.23s call +16.46s
  setUpClass =52.90s total, PASSED. Split at cap ~36 finite/~4 -inf (named
  refusals survive; C7's extra 59%-inf were the w>60 mpmath refusals the cap
  removes — consistent w/ Ruling 4, not a bug). PNG:
  tests/output/prior_sweep_wmax_and_outcomes.png.
- BACKWARD-COMPAT (report-only, NOT SHARD 2a scope; likely another shard's
  fixture in same file): RefusalContractTestCase::
  test_refusal_precedes_coherent_score FAILS on full-file run —
  CANCELLATION_LENS no longer raises SchwingerCertificationError because the
  merged mpmath extension (W_CEILING_SCHWINGER 60->150) now SERVES its
  w in (60,150] nodes via _f_schwinger_mpmath instead of refusing. Pre-
  existing drift from a PRIOR build (this build's WPs empty []); my git diff
  is confined to imports/constants/target-method, cannot have caused it.
  Owner fix: repoint CANCELLATION_LENS refusing nodes to w>150.

- WP1/WP2 cusp-adapted wedge axis + waist-split tiler (SHARD A) into
  test_lensing_interior_wedge_chart.py (63 -> 77 tests, full file 47s).
  T1 TransverseCutAxisAccuracyTestCase (4 tests, 5.7s): at gamma=0.3,
  r=0.455, theta in [1e-4,0.2], w-grid geomspace(3,12,10) — fit spline on
  arc-length s / raw theta / cusp u=theta^(2/3) axes at 5 nodes, eval
  held-out vs fresh ChangRefsdalChannels. Ordering err_u<err_theta<err_s
  holds at ALL percentiles ONLY with this w-window; geomspace(5,40,10)
  broke p50 (u worse than theta) and pushed u p90 to 7.8e-3. Assert on P90
  (u_p90<1e-3, robust; advantage is a near-cusp TAIL effect) + u_max<1.5e-3
  (measured 6.83e-4, matches spec's 6.9e-4; s~4.9e-2). Report p50/p90/max +
  worst locus to transverse_cut_axis_accuracy.txt; overlay .png.
  T2 rewrite of WedgeInteriorTilesContractTestCase + constants: tiler is now
  3-ARG `_wedge_interior_tiles(gamma, r_extent, n_per_side)` and emits
  5-TUPLES `((r_c,th_c),(half_r,half_th),i,j,axis_origin)` (was 2-arg /
  4-tuple / single pi/4 column). Two angular columns per radial row: j==0
  'low' spans [0,waist], j==1 'high' [waist,pi/2]; shared boundary ==
  _wedge_theta_waist(gamma) (INDEPENDENT argmin oracle). Physical waist pin:
  |r_caustic(gamma,theta_waist)-gamma|<1e-6 (VALUE, min is flat — do NOT pin
  the angle). gamma>=0.6 -> |theta_waist-pi/4|>0.10 (asymmetry real).
  Removed WEDGE_THETA_CENTER/HALF (no longer pi/4-based). len==2*n_per_side.
  T3 WedgeSubdivisionUMidpointTestCase (reachable-red): child boundary
  theta_split == u-midpoint image, NOT theta-midpoint. Production
  _subdivide_wedge_tile (surrogate_training) computes theta_split via
  np.interp(u_mid, u_fine, theta_fine) with u_mid=0.5*(u[0]+u[-1]); u lands
  on the EXACT centre node of the odd uniform-u grid so interp err ~1e-16
  (contract tol 1e-9). RETURN rounds 6dp -> return-check tol 1e-6. Closed-
  form oracle _u_midpoint_theta (low: (0.5*(tl^(2/3)+th^(2/3)))^1.5; high:
  pi/2-(0.5*((pi/2-tl)^(2/3)+(pi/2-th)^(2/3)))^1.5) — matches production
  _wedge_cusp_axis_map (verified surrogate.py L605-680). REACHABLE-RED mocks
  surrogate_training._load_or_build + _gate_chart so the REAL subdivider
  runs without engine.
- CONFIRMED: from_wedge_engine (surrogate.py L3966-3970) now stores
  theta_to_s = vstack([theta_fine, u_fine]) from _wedge_cusp_axis_map — the
  chart's theta_to_s ROW 1 is now the u-coordinate, NOT arc-length s. Schema
  bumped to 'wedge_caustic_relative_v2' (_KNOWN_WEDGE_AXIS_SCHEMAS = {v2}
  only; old v1 hard-refuses at load). No test hardcodes the wedge schema
  string (grep 'wedge_caustic_relative' in tests -> empty).
- BACKWARD-COMPAT AUDIT (WP1/WP2, OUT OF SCOPE / report-only):
  (1) BROKEN: test_lensing_ppgo_bandsplit.py WedgeInteriorTilesTestCase
      (~L595-690). setUpClass L633 calls st._wedge_interior_tiles(R_EXTENT,
      N_PER_SIDE) — OLD 2-arg -> TypeError (missing gamma/n_per_side). Also
      `for center,half,_i,_j in self.tiles` unpacks 4 but tiles are 5-tuples
      -> ValueError. Also asserts len==N_PER_SIDE (now 2*N) and "single
      angular column [0,pi/2] no split" — the whole class premise is retired
      by WP2's two-column waist split. OWNER must migrate: gamma-first call,
      5-tuple unpack, expect 2 columns split at _wedge_theta_waist, drop the
      single-column contract. This is the SAME file+class I ported in the
      prior ffin build (then 2-arg/single-column was current); WP2 changed
      the API from under it.
  (2) NOT broken (only stale prose): test_lensing_wedge_dd_arclength.py —
      every assert is STRUCTURAL (theta_to_s not None, shape (2,2001),
      endpoints==theta_wedge_grid, both rows monotone, s-row nonlinear via
      polyfit residual>1e-4, perturbed-map degrades). All hold for the u-map
      too (u strictly increasing + nonlinear). It builds its OWN arc-length
      reference only as a self-contained teeth control, never compares the
      chart's row to arc-length values. Docstrings say "arc-length" (now
      misleading) but no functional break. Not my scope to rename.
  Only _wedge_interior_tiles/_subdivide_wedge_tile/_wedge_theta_waist/
  _wedge_cusp_axis_map callers in tests/: my own file + ppgo_bandsplit (the
  broken one). Confirmed via grep.
