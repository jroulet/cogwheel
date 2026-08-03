# Test Dev Short-Term Observations

- Created `cogwheel/tests/test_lensing_extrapolate_floor.py` for WP1's
  `_extrapolate_floor` function in ppgo_map.py.  8 tests across 4 classes
  (PowerLawExtrapolation, NonPhysicalSlopeRefusal, PoorFitRefusal,
  SelfFalsification).  All green.  Key design note: the spec's
  geomspace(1,60,24) grid with bar=1e-4 hits BOTH the R² guard (0.84<0.9
  due to beat aliasing) and the MAX_RATIO guard (ratio~17>5); adapted to
  geomspace(10,2000,50) which fits cleanly (R²=0.91, ratio=0.53).
  Backward audit: WP1 is purely additive (_measure_cell signature unchanged);
  no existing tests broken. Bandsplit test uses stubbed error=0.0 so
  extrapolation never triggers. Ghost test mentions _measure_cell in docs
  only, doesn't call it.
- Extended the suite with 3 new classes (total 8): ExcessiveExtrapolation
  RefusalTestCase (fast, analytic), InteriorCellExtrapolationTestCase
  (engine-backed, TRAIN_TIER gated), ExteriorCellPreservationTestCase
  (engine-backed, TRAIN_TIER gated). 10 fast tests pass; 4 engine tests
  correctly skip without COGWHEEL_TRAIN_TIER.

- Extended `test_lensing_exterior_windows.py` with 2 new classes for WP1:
  `InteriorWnpdAccuracyTestCase` (4 tests) and `TrainingConfigWnpdFieldTestCase`
  (4 tests). All 8 green. Key finding: the Architect spec's falsification
  claim ("WNPD=6 fails the 0.05 bar at gamma=0.65") is FALSE at the existing
  smoke geometry (SACRC_S_RANGE, SACRC_D_RANGE, n_s=5, n_d=5) — measured eps is
  0.0002 even at WNPD=6 because the SACR-C envelope is dominated by spatial
  smoothness at this patch. Replaced with a wiring test + node-count test that
  proves the field is load-bearing (different node counts) and correctly wired.
  Added `_interior_chart_wnpd` (lru_cached) and `_wnpd_heldout_eps` helpers.
  The `_wnpd_heldout_eps` catches LensDomainError from `_evaluate_chart` (the
  wrap-into-arc refusal at gamma=0.40 seed=42 draw #5).


- Created `cogwheel/tests/test_lensing_fold_ppgo_correction.py` for
  WP1+WP2 fold_ppgo_correction. 14 tests in 4 classes
  (MonotoneImprovement, LargeXiNoOp, AxisAngleCorrection,
  SelfFalsification). All green in 8.3 s.
  Key findings: (1) The spec's monotone-improvement element-wise claim
  does NOT hold against the full wave operator at w>25 — at higher w the
  diffractive error drops below the Airy residual. Adapted test to use
  w=5..15 where fold divergence dominates and monotone holds. (2) The
  spec's "large-xi no-op at rho=3.5" actually triggers a structural
  fallback (b3=0 at axis angles -> _fold_amplitudes=None -> byte-identical
  fallback). Tested as byte-identical no-op. (3) The 7% correction is
  measured as |corrected - raw|/|raw| at high w (geometric limit); it
  oscillates between 4-40% with carrier-phase interference. (4)
  Schwinger ceiling at w=60 prevents using ChangRefsdalChannels as oracle
  for w>=100 near the caustic.
  BACKWARD-COMPAT FINDING: test_lensing_ppgo_bandsplit.py
  TruncationOnRefusalTestCase (2 tests) FAILS because WP2 replaced
  geometric_amplification with fold_ppgo_correction in _measure_cell,
  but the test's mock only patches geometric_amplification. The
  fold_ppgo_correction does real geometry work and may not fall back.
  Fix needed: additionally patch _airy_fold.fold_ppgo_correction (or
  patch it as the ppgo entry point in ppgo_map). NOT fixed here —
  owned by a separate Test Dev run per scope discipline.

- Extended `test_lensing_fold_ppgo_correction.py` with 2 new classes
  (total 6 + SelfFalsification = 7 classes, 23 tests):
  `UniformErrorEstimateRelaxationTestCase` (4 tests: xi=0 returns 0.0,
  xi=-1.0 returns None, xi=1.0 returns finite positive, small-xi
  continuity) and `FallbackIdentityTestCase` (5 tests: macro-saddle
  byte-identical, degenerate-b3 axis byte-identical, far-exterior axis
  byte-identical, scalar-input array-path byte-identical, non-fallback
  teeth). All 23 tests green in 4.65s.
  Key finding: scalar-input fold_ppgo_correction vs scalar-input
  geometric_amplification differ by ~1 ULP (5e-17j) due to different
  FP reduction order in the scalar vs array code paths. The byte-identity
  guarantee holds against the ARRAY-path extraction (which is what
  fold_ppgo_correction's internal _fallback() actually computes).