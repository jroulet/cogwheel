# Test Dev Short-Term Observations

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
