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
