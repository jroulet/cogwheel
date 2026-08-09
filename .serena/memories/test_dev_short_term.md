# Test Dev Short-Term Observations

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
