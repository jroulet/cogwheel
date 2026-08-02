# Professor Short-Term Observations

## C8 fence retirement review (2026-08-01)

- **test_lensing_born.py**: 53/53 pass (25s). ExteriorFenceTestCase,
  SaddleExteriorFenceTestCase, all SADDLE_FENCE_* constants, ANNULUS_INNER_RADIUS,
  GAMMA_FENCE, saddle_caustic_max_y fully deleted from tests AND production code.
  Zero stale references (grep-verified). C8FenceRetirementTestCase correctly
  demonstrates gamma=0.80, 0.90, 1.04 admitted; parity-wall (gamma=0.998) still
  refuses; large-w guard A still refuses.
- **test_lensing_ppgo_map.py**: 37/37 pass (9.5s). CausticRhoByteEquivalenceTestCase
  confirms the rename from annulus_rho -> caustic_rho is purely cosmetic (exact
  byte-equal, not almostEqual). No `annulus_rho` references remain anywhere.
- **CausticRelativeClassificationTestCase**: tests both parities (pos gamma=0.5 at
  |y|=2.0 -> rho=1.414 > 1 -> 'born'; saddle gamma=1.3 at |y|=2.5 -> rho=1.458 > 1
  -> 'born'; interiors rho < 1 correctly NOT 'born'). Parity-independent threshold
  confirmed.
- **Physics sanity**: caustic_rho = |y|/caustic_reach. For positive parity gamma=0.5:
  reach = sqrt(2) ≈ 1.414 (from axis-aligned astroid cusp). For saddle gamma=1.3:
  reach ≈ 1.714 (deltoid cusp radius). Both verified numerically correct.
- **Architecture**: surrogate_census.classify_fallthrough calls the ONE authoritative
  `caustic_rho` from ppgo_map; the 'born' classification is rho > 1 on both parities.
  No special-casing or gamma-guards beyond the parity wall margin (DELTA_GAMMA_P).
- **BornCensusReachableRedTestCase._classify**: uses `mock.patch('...caustic_rho',
  lambda gamma, abs_y, kappa=0.0: 0.0)` to disable born branch — forcing rho < 1
  (interior). Confirmed the reachable-red foil works: same draw flips to 'out-of-box'.
- **SelfFalsificationTestCase**: gamma=0.998 (pos parity, |0.998-1|=0.002 ≤ 0.005);
  SaddleSelfFalsificationTestCase: gamma=1.003 (saddle, |1.003-1|=0.003 ≤ 0.005).
  Both correctly refuse via parity-wall margin.
- **CENSUS_NONANNULUS_Y1_EIG = 0.5** (was 2.0): at gamma=0.45, reach ≈ 1.214,
  rho = 0.5/1.214 ≈ 0.41 < 1 → interior (correct).
  **SADDLE_CENSUS_NONANNULUS_Y1_EIG = 1.0** (was 2.0): at gamma=1.2, rho < 1 → interior.
- Verdict: PASS. Heavy full-sampling validation is operator-deferred.
