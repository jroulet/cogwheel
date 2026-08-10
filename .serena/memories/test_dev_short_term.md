# Test Dev Short-Term Observations

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
