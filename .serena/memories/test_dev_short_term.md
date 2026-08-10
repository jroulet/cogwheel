# Test Dev Short-Term Observations

2026-08-10: Extended test_lensing_exterior_polar_fold.py (TEST 7/8, NaN-hole filling + self-falsification diagnostics, 6 new tests, 58 total):
  - FoldCarrierSelfFalsificationTestCase (6 tests, was 4): added chart_none +
    off-grid probe points; test_zero_carrier_off_grid_ratio (err_none/err_correct >
    SELF_FALSIFICATION_MARGIN at off-grid (rho,theta_c)); diagnostic grouped bar
    chart (log10(eps) for correct/wrong/none at on-grid and off-grid, with reference
    lines for NODE_EXACT_TOL and SELF_FALSIFICATION_MARGIN).
  - FoldCarrierFromEngineTestCase (12 tests, was 8): test_rho_u_carrier_not_all_zeros;
    test_ghost_kernel_delay_matches_carrier_at_valid_nodes (independent ghost_kernel
    probe with MEDIAN across gamma-band matches stored rho_u_carrier to 1e-12);
    test_filled_nan_nodes_have_smooth_derivatives (NaN-filled nodes identified by
    GhostDomainError at all band gammas; rho-derivative ratio vs neighbor cols < 2,
    theta_c-derivative ratio vs neighbor rows < 2; handles boundary fills on 4×4
    grid); test_ghost_boundary_diagnostic_heatmap (rho_u_carrier colormap with red
    X at NaN-filled nodes, green o at ghost-valid nodes).
  KEY FIX: ghost_kernel delay match uses MEDIAN across all gamma-band gammas
  (production _compute_rho_u_carrier stores median) — not first-match.
  KEY FIX: import path for geometry is `cogwheel.lensing.chang_refsdal.geometry`
  (not `cogwheel.lensing.geometry`).
  All 58 green. No regression on test_lensing_exterior_carrier.py (23) or
  test_lensing_exterior_admission.py (76, 1 pre-existing skip).

2026-08-10: Extended test_lensing_surrogate_training.py (fold-carrier training), 6 new classes / ~26 tests):

  NEW CONSTANTS: _FOLDCARRIER_N_PER_SIDE=2, _FOLDCARRIER_N_GAMMA=4,
  _FOLDCARRIER_N_RHO=4, _FOLDCARRIER_N_THETA_C=3 (bumped to 4 in
  from_engine calls due to _uniform_axis requiring >=4),
  _FOLDCARRIER_M_LENS_MSUN=(10,15), ghost-zone bounds, FAR_RHO=3.2.

  New classes:
  - FoldCarrierNeedsGhostTestCase (5 tests): unit test of
    _needs_fold_carrier — mock geometry.ghost_kernel to return valid
    GhostContribution (True), GhostDomainError at all corners (False),
    GhostDomainError at some/some (True), gamma_band multi-gamma probe,
    _from_caustic_fixed/macro_matrix exceptions silently skipped.
  - FoldCarrierNeedsGhostSelfFalsificationTestCase (3 tests): always-raise
    vs always-succeed differ; gamma_band is load-bearing (without band
    only mid-gamma probed).
  - FoldCarrierContinuitySafetyNetTestCase (7 tests): synthetic 4D grid
    with sharp jump along gamma/rho axes raises CarrierDiscontinuityError;
    uniform/zero envelopes pass; single-node axes skipped; from_engine
    propagates injected CarrierDiscontinuityError (mocked
    _assert_exterior_polar_carrier_continuity); fold_carrier=False path
    succeeds (engine-backed, gated).
  - FoldCarrierContinuitySelfFalsificationTestCase (3 tests): smooth
    does not raise; sharp jump does raise; rho_carrier=None fallback
    succeeds on far tile.
  - FoldCarrierTrainingIntegrationTestCase (~8 tests, TRAIN_TIER):
    engine-backed integration — train(regions=('exterior',),
    m_lens_range=(10,15)).  Ghost-zone tiles have rho_carrier not None;
    far tiles may have None; ghost_drop_count=0; rho_carrier+rho_log_axis
    compose; charts serve finite values; rho_carrier shape matches
    rho_grid; rho_carrier elements finite.
  - FoldCarrierTrainingIntegrationSelfFalsificationTestCase (2 tests,
    TRAIN_TIER): shifted zone criteria exclude all; nonzero ghost drop
    can fail.

  PRE-EXISTING FIX: _RHO_LOG_SCHEMA_V3 →
  surrogate_module._EXTERIOR_POLAR_AXIS_SCHEMA_V4 (V3 constant was
  renamed in the fold-carrier schema bump).

  All fast-tier tests green (100 pass, 67 TRAIN_TIER skip).
  No regression on existing non-TRAIN_TIER tests.



2026-08-10: Extended test_lensing_surrogate_training.py (DT-8, ghost_excluded_tiles in region report):
  - GhostExcludedTilesInRegionReportTestCase (3 tests): calls the real
    _train_band_charts with mocked _exclude_ghost_dominated (always-True
    yields ghost_excluded_tiles=3 > 0; always-False yields 0); mocked
    _load_or_build to avoid engine chart building. Verifies key exists,
    is positive int, and counter respects the gate function's return.
  - GhostExcludedTilesInRegionReportSelfFalsificationTestCase (3 tests):
    zero count fails > 0 assertion; missing key fails In check; non-int
    values fail isinstance. 6 new tests, all green.
  CONFIG: n_gamma=4, n_rho=4, n_theta_c=4, n_farfield_tiles_per_side=2,
  gamma_mid=0.5, gamma_band=(0.46, 0.54).

2026-08-10: Extended test_lensing_exterior_polar_fold.py (TEST 4/5/6, 10 new tests, 52 total):
  - RhoCarrierCompositionTestCase (5 tests, was 1): COMPOSITION_HELDOUT_BAR=4e-3,
    ~81 off-grid probes.  eps within 4e-3; phase error <1e-3 rad at w=25;
    raw-rho-not-log-rho remodulation proof (chart with tau_c=slope*rho,
    rho_log_axis=True: phase_served matches w*slope*rho not
    w*slope*log(rho-1)); magnitude invariant (compares served with/without
    carrier_rate, both share spline-on-log-rho error budget, Δmag < 5e-13);
    phase diagnostic scatter plot.
    KEY FIX: angular distance uses abs(np.angle(np.exp(1j*(p1-p2)))) —
    naive abs(diff) with 2π-diff wrapping fails when diff > 2π.
    Magnitude test compares with-k vs without-k (same rho_log_axis) not
    analytic oracle — cubic-spline-on-log interpolation error (~1e-3)
    dwarfs the pure-phase rotation error (~5e-13).
  - FoldCarrierNpzRoundTripTestCase (6 tests, was 5): added
    test_loaded_vs_source_histogram_diagnostic (histogram of |ΔE| over
    all w-nodes, max|Δ|=0).
  - ExteriorPolar1DArtifactBackwardCompatTestCase (5 tests, NEW):
    V4 schema 1D rho_carrier loads, broadcasts to 2D, serves identically.
    Uses CONSTANT-IN-u carrier (broadcast of rho_u_carrier[:,0]) because
    the byte-identical claim requires no u-variation.
    Tests: load, shape, broadcast equality, byte-identical serve, heatmap.
  All 52 green. No regression on test_lensing_exterior_carrier.py (23 pass).

2026-08-10: Ported test_lensing_exterior_polar_fold.py from 1D rho_carrier to 2D rho_u_carrier API (WP-1: 2D fold-carrier on ExteriorPolarChart):
  Rewrote entire file for new production API (rho_carrier → rho_u_carrier,
  shape (n_rho, n_theta_c)), schema tag V4→V5
  (exterior_polar_rho_log_carrier_v1 → exterior_polar_rho_u_carrier_v2),
  NPZ key chart0_rho_carrier → chart0_rho_u_carrier.
  8 test classes, 42 tests, all green (no skips).

  KEY MEASUREMENT: theta_to_u introduces piecewise-linear interpolation
  error for the nonlinear theta_c→u mapping at 4 nodes (~22% in u),
  making off-grid accuracy ~37× worse than raw-theta axis.  For smoke-scale
  tests, avoid theta_to_u and use carrier bilinear in (rho, theta_c)
  directly.  The cusp-adapted u-coordinate accuracy is a separate concern
  from the carrier demodulation mechanism.

  Classes:
  - RhoCarrierNodeRoundTripTestCase (3 tests): 2D node-exact round-trip
    to 6.1e-14 with diagnostic scatter.  Backward-compat None-carrier.
    Storage wiring.
  - RhoCarrierContinuityGuardTestCase (3 tests): 2D-demodulated passes
    all axes (gamma, rho, theta_c); raw raises CarrierDiscontinuityError.
    Diagnostic bar chart.
  - RhoCarrierOffGridPhaseTestCase (4 tests): off-grid phase ≤1e-3 rad;
    magnitude invariant; residual phase span ≤1.63 rad (vs >3 rad raw);
    side-by-side diagnostic plot.
  - RhoCarrierCompositionTestCase (2 tests): rho_u_carrier + carrier_rate
    + rho_log_axis within 5e-2 bar.
  - FoldCarrierSelfFalsificationTestCase (4 tests): wrong carrier > bar
    and > 10× correct; sanity assert correct < bar; magnitude teeth.
  - FoldCarrierNpzRoundTripTestCase (5 tests): schema V5, byte-identical
    rho_u_carrier, carrier_rate/rho_log_axis preserved, served values
    bit-identical after round-trip.
  - FoldCarrierLegacySchemaHardRefusalTestCase (4 tests): V3/demod_v2
    hard-refuse; missing axis_schema raises ValueError; V5 accepted.
  - FoldCarrierMissingKeyBackwardCompatTestCase (2 tests): absent
    rho_u_carrier key loads as None; None-chart round-trips unmodified.
  - FoldCarrierFromEngineTestCase (8 tests): fold_carrier=True →
    rho_u_carrier not None, shape=(n_rho, n_theta_c), finite, KERNEL_SUM
    definition; heldout eps < 1e-2 node, 5e-2 off-grid; can serve.
  - FoldCarrierFromEngineBackwardCompatTestCase (4 tests):
    fold_carrier=False → rho_u_carrier=None, axes correct, finite
    carrier_rate, can serve.
  - FoldCarrierFromEngineSelfFalsificationTestCase (4 tests):
    True/has, False/lacks, True≠False, not-all-zeros.

  CONFIG: n_gamma=4, n_rho=4, n_theta_c=4, n_w=4, rho=[1.3,2.1],
  theta_c=[0.1,0.7], log_w=[ln(5),ln(30)], coeff_rho=2.5,
  coeff_u=-1.45 (bilinear tau_c(rho,theta_c)), from_engine
  (0.3,0.7)x(1.3,2.0)x(0,0.5)x(10,30) WNPD=6.

  No regression on test_lensing_exterior_carrier.py (23 pass).

2026-08-10: Extended test_lensing_exterior_polar_fold.py (WP fold_carrier_chart + fold_carrier_training, 41 total / 26 new tests):
  CONFIG — same as existing 4×4×4×4 grid, plus new from_engine constants
  _FROM_ENGINE_GAMMA_RANGE=(0.3,0.7), _FROM_ENGINE_RHO_RANGE=(1.3,2.0),
  _FROM_ENGINE_THETA_C_RANGE=(0.0,0.5), _FROM_ENGINE_W_RANGE=(10,30),
  n_gamma=4, n_rho=4, n_theta_c=4 (bumped from spec's 2 because
  _validate_axis requires >=4), WNPD=6, NODE_HELDOUT_BAR=1e-2,
  OFFGRID_HELDOUT_BAR=5e-2.

  New classes added (6 new, 26 new tests):
  - FoldCarrierNpzRoundTripTestCase (5 tests): schema tag is
    'exterior_polar_rho_log_carrier_v1', rho_carrier byte-identical
    after round-trip, carrier_rate & rho_log_axis preserved, served
    values match source after round-trip.
  - FoldCarrierLegacySchemaHardRefusalTestCase (3 tests):
    'exterior_polar_rho_log_v3' hard-refuses with ValueError,
    absent axis_schema raises ValueError, 'exterior_polar_carrier_demod_v2'
    also hard-refuses.
  - FoldCarrierMissingKeyBackwardCompatTestCase (2 tests): new-tag NPZ
    without chart0_rho_carrier key loads with rho_carrier=None;
    None-chart round-trips through NPZ with bit-identical served values.
  - FoldCarrierFromEngineTestCase (8 tests): from_engine(fold_carrier=True)
    produces rho_carrier is not None (shape=4, finite), carrier_rate
    finite, envelope_definition==KERNEL_SUM (not MINUS_GHOST), node-exact
    eps within 1e-2, off-grid eps within 5e-2, evaluate_chart returns
    finite.
  - FoldCarrierFromEngineBackwardCompatTestCase (4 tests):
    fold_carrier=False → rho_carrier=None, axes match expected counts,
    carrier_rate finite, chart serves finite values.
  - FoldCarrierFromEngineSelfFalsificationTestCase (4 tests): True/has,
    False/lacks, True vs False served DIFFER at off-grid rho (>1e-6),
    rho_carrier is not all zeros.
  All 41 green. No regression on test_lensing_exterior_carrier.py (23 pass).

  PRE-EXISTING FAILURES in test_lensing_surrogate.py (StaleSchema + RhoLogAxis
  Serialization tests, 6 fails + 6 errors): use retired
  'exterior_polar_rho_log_v3' schema — unrelated, coder's memory flagged
  as needing separate test_dev fix.

2026-08-10: Wrote test_lensing_exterior_polar_fold.py (15 tests, 5 classes):
  - RhoCarrierNodeRoundTripTestCase (3 tests): node-exact round-trip to 5e-13
    (max measured 1.33e-13); backward-compat None rho_carrier; storage wiring.
  - RhoCarrierContinuityGuardTestCase (2 tests): raw oscillating envelope
    raises CarrierDiscontinuityError; demodulated passes.
  - RhoCarrierOffGridPhaseTestCase (3 tests): off-grid phase < 1e-3 rad at
    w=25 (measured ~7e-15); magnitude invariant under remodulation (measured
    < 2e-13); diagnostic plot at output/exterior_polar_fold_offgrid_phase.png.
  - RhoCarrierCompositionTestCase (3 tests): rho_carrier+carrier_rate+
    rho_log_axis composite within bar (5e-2); remodulation uses RAW rho not
    log(rho-1) (verified via known linear rho_carrier + phase comparison);
    diagnostic plot at output/exterior_polar_fold_composition.png.
  - FoldCarrierSelfFalsificationTestCase (4 tests): wrong rho_carrier (+0.3
    rad offset) > 5e-13; wrong/correct ratio > 10×; correct chart within bar
    (sanity); magnitude invariance holds even with wrong phase (proves phase
    error is phase error, not amplitude error).
  Config: 4×4×4×4 grid (w, gamma, rho, theta_c), rho[1.3..2.1],
  tau_c 0→16 rad, gamma[0.35..0.65], log w[ln(5)..ln(30)].
  No regression on test_lensing_exterior_carrier.py (23 tests pass).

2026-08-10: Extended test_lensing_exterior_admission.py (DT-7, farfield_exterior_tiles ghost integration):
  - FarfieldExteriorTilesGhostExclusionTestCase (7 tests): builds
    _farfield_exterior_tiles with/without ghost kwargs (gamma=0.5,
    gamma_band=(0.48,0.52), ghost_drop_count); ghost exclusion drops 12
    of ~33 inspected tiles, 4 of which would have been admitted (net
    drop 4); ghost_drop_count=12 >> net_dropped=4 > 0; near-axis tiles
    dropped; far-outer tiles retained; backward-compat gamma=None
    unfiltered.  N_PER_SIDE=8, SOURCE_MAG_MAX=8.0.
  - FarfieldExteriorTilesGhostExclusionSelfFalsificationTestCase
    (4 tests): threshold-zero inert, mock _exclude_ghost_dominated→False
    restores unfiltered, threshold-zero ghost_ct=0, real ghost exclusion
    makes filtered≠unfiltered.
  2 new classes, 11 new tests (10 green, 0 skip).  87 total (1
  pre-existing skip: CuspAlignmentSelfFalsificationTestCase).

  MEASURED: ghost_drop_count=12 (all ghost-inspected tiles), net_dropped=4
  (tiles that would have been admitted but are ghost-excluded).  The 8=
  12-4 tiles fail BOTH ghost exclusion AND admission (they're too close
  to the caustic).

2026-08-10: Extended test_lensing_exterior_admission.py (DT-4/DT-5/DT-6):
  - DecayGateOnlyExclusionTestCase (4 tests): single-gamma (no gamma_band)
    correctly excludes a near-axis tile (rho=1.1, theta=0.05) with Im(tau_c)<0.4
    at surviving corners, and retains a higher-rho tile (5.16) with Im(tau_c)>=0.42;
    quantitative corner-Im assertions confirming the exclusion/retention is
    physically grounded.
  - MultiGammaBandEdgeTestCase (3 tests): tile (rho=5.16, theta=0.05) passes at
    gamma_mid=0.5 (without gamma_band → False) but fails at gamma_hi=0.54
    (with gamma_band → True); quantitative corner-Im assertion confirms
    band-edge probe triggers.
  - CenterProbeStraddleTestCase (3 tests): no physical tile found where corners
    all >= 0.4 and center < 0.4 (low-rho corners always closer to caustic);
    mock-based proof that center probe is load-bearing — mocking bad center
    flips result on an otherwise-retained tile; corners-only confirmation.
  - DecayGateSelfFalsificationTestCase (4 tests): threshold-zero unexcludes,
    retain assertion teeth, single-gamma assertion teeth, center-mock
    inversion sanity check.
  4 classes, 14 tests, all green; 76 total (1 pre-existing skip).

2026-08-10: Updated test_lensing_surrogate.py for carrier_demod schema migration (carrier_demod_v2):
  - ExteriorPolarStaleSchemaHardRefusalTestCase: 8 tests (was 3). Both
    exterior_polar_rho_theta_c and exterior_polar_rho_u_v1 hard-refuse;
    new carrier_demod_v2 schema accepted; carrier_rate preserved through NPZ
    round-trip (0.5); zero-carrier backward compat (missing key → 0.0);
    NaN/Inf guard. Updated _build_minimal_npz to accept carrier_rate and
    include_carrier_rate params.
  - ExteriorPolarCuspAdaptedFromValuesTestCase: 6 tests (was 4, +2 for
    carrier_rate default=0.0 and storage when nonzero).
  - ExteriorPolarCuspAdaptedSerializationTestCase: 3 tests (was 2, +1 for
    carrier_rate surviving production save/load round-trip).
  All 26 ExteriorPolar tests green.
  PRE-EXISTING FAILURES (from production code's from_engine carrier
  estimation): BetaEliminationTestCase (2 tests, eps=0.144 >> 2e-9) and
  LnlikeAccuracyTestCase.test_positive_served_lnlike_tracks_engine (0.582 >
  0.5). These pass on HEAD (old production code) — the carrier_demod
  production change from from_engine modifies the chart's trained envelope
  (demodulates with estimated k_chart) and the reference/envelope comparison
  at serve time no longer matches at the old tolerance.

2026-08-10: Extended test_lensing_exterior_carrier.py (rho_log_axis A/B + composition):
  - RhoLogAxisABComparisonTestCase (4 tests): builds chart pairs from identical
    grids/values, rho_log_axis=True vs False. Log-axis eps is strictly smaller
    (ratio > 3×) for both positive (gamma=0.5) and saddle (gamma=1.5) parity.
    Envelope is (rho-1)^{-0.5} * (1 + A*(w/w_mean-1)) — power-law radial
    singularity that cubic spline resolves better in ur=log(rho-1). RHO_GRID_LOG
    = [1.05, 1.15, 1.30, 1.50] (4 nodes). Off-grid probes: 3 random per
    inter-node interval. Diagnostic plot at output/.
  - RhoLogCarrierCompositionTestCase (2 tests): rho_log_axis=True +
    carrier_rate=0.05 compose correctly — off-grid eps < 5e-2 bar.
    Diagnostic plot at output/.
  - RhoLogAxisABSelfFalsificationTestCase (2 tests): flat (constant-rho)
    envelope has no improvement (ratio < 3×), proving the A/B detector has
    teeth. Synthetic assertion-can-fail test.
  - RhoLogCompositionSelfFalsificationTestCase (2 tests): wrong
    carrier_rate (Δk=0.1) and zero carrier_rate for modulated envelope both
    exceed the held-out bar, proving the composition detector has teeth.
  4 new classes, 10 new tests (23 total), all green.
  PRODUCTION BUG FOUND: surrogate.py:2775 uses math.log() without importing
  math — crashes all rho_log_axis=True serve paths with NameError.
  Workaround in test: sg.math = math injected at module level.
  Helper fixes: _build_rho_dependent_chart and _envelope_exact had
  float64→complex128 in-place casting errors (envelope *= exp(i*k*w) on
  float array) — fixed with (envelope + 0j) pattern.

2026-08-10: Wrote test_lensing_exterior_carrier.py (carrier demodulation round-trip):
  - NodeRoundTripTestCase (3 tests): node-exact round-trip to 1e-13 at all 64 (4³) spatial nodes × 5 w-nodes, backward-compat zero-carrier, wiring guard (carrier_rate is stored correctly).  Measured node-err ≤ 3e-15.
  - HeldOutAccuracyTestCase (2 tests): w-midpoint eps 3.12e-05 < 1e-3 bar (5 w-nodes, 20% amplitude modulation), diagnostic plot saved to tests/output/.
  - SelfFalsificationTestCase (4 tests): Δk=0.1 corrupted eps=1.06e-2 > bar (> 340× correct), zero-k eps=1.87e-3 > bar (> 60× correct), sanity assert correct < bar.
  - CarrierSelfFalsificationTestCase (3 tests): wrong reference detectable, node-exact assertion can fail deliberately, coarse 4-node grid with large modulation fails bar (5.57e-3 > 1e-3).
  Key bug found and fixed: _build_w_envelope originally computed w_mean from the QUERY grid, not the TRAINING grid — held-out reference used different w_mean causing inflated eps (7e-3 vs real 3e-5).  Fixed by adding optional w_mean parameter, passing _W_MEAN_TRAIN for all held-out references.
  5 classes, 12 tests, all green.  Diagnostic plot: exterior_carrier_held_out_residual.png.

2026-08-10: Wrote ghost-dominated tile exclusion tests in test_lensing_exterior_admission.py:
  - GhostDominatedExclusionTestCase (2 tests): DT-1 near-axis tile where Im(tau_c)~0.013
    at corners triggers exclusion (True); gamma_band=None also triggers (center gamma
    0.5 has Im~0.013 too); threshold-zero patch makes gate inert (False).
  - GhostDominatedKnownGoodTestCase (2 tests): DT-2 far-off-axis tile retained (False);
    measured min Im(tau_c)=5.36 >> 0.4 across corners at both band edges.
  - GhostDominatedNoGhostTestCase (1 test): DT-3 saddle-parity tile at gamma=1.5,
    rho=4.0, theta_c=1.4 where ALL corners raise GhostDomainError — returns False
    (retainable, ghost-free KERNEL_SUM).
  - GhostDominatedSelfFalsificationTestCase (3 tests): threshold-zero, saddle-parity
    tile with existing ghost below threshold at (1.3, 0.175) returns True (proves
    DT-3's False is not blanket saddle-retain), known-good assertion can be falsified.
  All 8 new tests green; 61 total (1 pre-existing skip). Imported channels module
  for _GHOST_DECAY_IM_THRESHOLD mocking. No existing test breakage.

2026-08-09: Wrote test_lensing_exterior_admission.py WP3 (exterior-cusp-exclusion):
  - ExcludeNearCuspBandEdgeTestCase (2 tests): gamma_band band-edge check is load-bearing
    (excluded at gamma_lo=0.25 where r_caustic is SMALLER, NOT at mid-gamma=0.30;
    gamma_band=None returns False).  Measured fixture: rho=1.15, θ=20°, half=0.04, d_ex=0.15.
  - DeltoidCuspSourceAnglesTestCase (4 tests): saddle gamma=1.5 D₂-folded angles in [0,π/2],
    sorted, deduplicated (round to 12dp to merge float-identical angles), includes off-axis
    angle, differs from _cusp_source_angles at same gamma, empty for positive parity (gamma=0.5).
  - FarfieldTilesCuspExclusionTestCase (3 tests): filtered tiles are strict subset, every
    excluded tile has corner within _CUSP_EXCLUSION_DISTANCE of cusp at band-gamma, backward-compat.
  - ExteriorCuspExclusionSelfFalsificationTestCase (3 tests): explicit d_exclude=0 clears,
    mock _exclude_near_cusp -> False restores unfiltered, positive parity returns empty.
  Spec corrections: (a) the spec's rho=1.3, theta=π/8, d_ex=0.5 excludes at ALL gammas
  (corner within d_ex even at mid) — measured working params rho=1.15, θ=20°, half=0.04,
  d_ex=0.15.  (b) exclusion happens at gamma_lo (r_caustic smaller), not gamma_hi as spec
  claims.  (c) _farfield_tiles has no d_exclude param — used mock.patch for self-falsification.
  5 classes, 12 tests, all green. by Dreamer on 2026-08-09)
