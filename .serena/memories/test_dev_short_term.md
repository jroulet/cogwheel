# Test Dev Short-Term Observations

## C10 build: test_lensing_born.py — fence retirement verification (no changes needed)

- Suite already fully updated in C8 build; all 53 tests pass in ~25 s.
- Backward-compat audit confirmed: no stale references to GAMMA_FENCE,
  annulus_rho, saddle_fence, or saddle_caustic_max_y in any test file.
- All spec requirements already implemented: ExteriorFenceTestCase/
  SaddleExteriorFenceTestCase deleted, fence constants removed,
  BornCensusReachableRedTestCase uses caustic_rho mock, C8FenceRetirementTestCase
  and CausticRelativeClassificationTestCase present and passing.
- Pyright type warnings are all duck-typing false-positives (SimpleNamespace
  as surrogate, **kwargs dict unpacking) — runtime correct.


## C9 build: test_lensing_ppgo_map.py — caustic_rho rename verification

- Suite already fully ported to `caustic_rho` (was `CausticRhoByteEquivalenceTestCase`).
- 37 tests pass in ~10 s; no modifications needed — the rename was purely cosmetic.
- Backward-compatibility audit: `annulus_rho` absent from entire `cogwheel/tests/`
  except the descriptive class name `OuterAnnulusRhoCapTestCase` in
  `test_lensing_ppgo_bandsplit.py` (doesn't call the old function, still green).
- No fence/classify_fallthrough references in this file (those are other suites' domain).


## C8 build: test_lensing_born.py fence retirement

- Deleted: ExteriorFenceTestCase, SaddleExteriorFenceTestCase, _plot_astroid,
  _plot_saddle_fence, _saddle_off_on_axis, all SADDLE_FENCE_* constants,
  FENCE_SERVE/REFUSE/ABSY/ASTROID_TOL constants.
- Updated CENSUS_NONANNULUS_Y1_EIG from 2.0 to 0.5 (caustic reach ~ 1.214,
  needs rho < 1 for interior); SADDLE_CENSUS_NONANNULUS_Y1_EIG from 2.0 to 1.0.
- BornCensusReachableRedTestCase: mock.patch on GAMMA_FENCE -> caustic_rho mock.
- SaddleCensusReachableRedTestCase/SelfFalsification: saddle_caustic_max_y mock
  -> caustic_rho mock returning 0.0.
- SelfFalsificationTestCase: test_refuse_gamma_actually_raises (gamma=0.80) ->
  test_parity_wall_margin_actually_raises (gamma=0.998).
- SaddleSelfFalsificationTestCase: removed fence test, added wall-margin test.
- New C8FenceRetirementTestCase: gammas 0.80, 0.90, 1.04 now pass born_gate.
- New CausticRelativeClassificationTestCase: classify_fallthrough uses rho > 1
  on both parities (positive gamma=0.5, saddle gamma=1.3).
- Key insight: the old annulus inner radius (3.0) is no longer the boundary;
  caustic_rho = |y| / caustic_reach > 1 is the new one-for-both-parities gate.
