# Test Dev Short-Term Observations

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
