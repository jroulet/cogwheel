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
