# Build 8h-b6 — Close the three admission/tiling defects before any retrain

## Mission

Build 8h-b4's exterior admission repair is verified working (coverage
0.000 -> 0.9725 in the gamma 0.80-0.90 band, zero false admits). Test
work then localized THREE production defects in the same
admission/tiling path. Fix them BEFORE the next training run: every one
of them changes which charts get built or registered, so retraining
first would mean retraining twice.

1. **Exterior tiler does not cusp-align `theta_c` (accuracy defect).**
   `_farfield_interior_tiles` aligns tile edges to the astroid cusp
   rays; `_farfield_exterior_tiles` deliberately lays a UNIFORM
   `theta_c` grid on the premise that "the exterior fit does not need
   it; the defect is zero-tile admission, not eps loss". **That premise
   is contradicted by measurement.** The caustic-fixed exterior
   coordinate is anchored on `r_caustic(gamma, theta_c)`, which has a
   slope KINK at each cusp ray, so a tile straddling one carries a
   kinked coordinate map. Measured at (gamma=0.40, rho=2.183): held-out
   eps is **1.5e-4 off the cusp ray and 2.6e-1 ON it**; sweeping gamma
   at fixed off-ray theta_c gives 1.0e-4 -> 3.1e-4, versus 1.6e-1 ->
   3.9e-1 on-ray (growing with gamma as the cusp sharpens). This is the
   sole cause of the one deliberately-red test in
   `test_lensing_surrogate.py::EnvelopeReconstructionTestCase`
   (positive-box eps 2.613e-1 vs a 0.2 budget-calibrated tolerance;
   12 of 13 held-out points pass, median 3.3e-2). Align exterior tile
   boundaries to the four astroid cusp rays, mirroring the interior
   tiler. DO NOT widen the tolerance — the fix must make the measured
   eps drop, and that red test is the acceptance.
2. **`admits_exterior` discards a whole column for partial out-of-box
   probes (coverage defect).** The per-column PREDICATE recovers ~1.000
   point coverage, but at the production `n_farfield_tiles_per_side =
   5` (72-degree-wide columns) TILE-level coverage of the 0.80-0.90
   band is only **0.56**; disabling only the `source_magnitude_max`
   test raises it to **0.88**. A chart node outside the prior box is
   WASTEFUL, not WRONG — rejecting the whole column throws away ~1/3 of
   the coverage the repair just recovered. This is the same
   over-conservatism class WP1 fixed, one order milder. Admit on the
   in-box portion (e.g. clip the column's rho span to the box, or admit
   when a sufficient fraction of probes are in-box) while keeping the
   caustic-distance invariant EXACT.
3. **`_InteriorAdmission.admits` can false-admit by cloud
   discretization (correctness defect).** At band (0.45,0.55),
   theta=0, rho=0.74 the EXACT `nearest_caustic_point` distance is
   0.0462 < `eta_max` = 0.05, yet production admits, because the
   200-point caustic cloud reads farther. Slop ~8% of `eta_max`. The
   exterior invariant is unaffected (margin 0.35), so this is
   interior-only. Fix by densifying the cloud, adding a margin, or
   testing against the exact oracle at the decision boundary —
   whichever the Professor judges cheapest for the accuracy gained.

## Measured facts (pre-answered — do not re-derive)

- Verified new exterior coverage: 0.9960 / 0.9936 / 0.9864 / 0.9798 /
  0.9725 across bands (0.30,0.40) (0.40,0.50) (0.50,0.70) (0.70,0.80)
  (0.80,0.90); old rule gave 0.944 / -- / 0.632 / 0.271 / 0.000.
  False-admit margin: min exact caustic distance over all admitted-tile
  probes 0.45/0.42/0.38/0.35 vs eta_max 0.05.
- The acceptance suite already exists and is green:
  `cogwheel/tests/test_lensing_exterior_admission.py` (23 passed),
  covering all four bands, the exact-zero-false-admit invariant, and a
  two-sided reachable-red. EXTEND it; do not re-author it.
- Interior tiler already implements cusp-ray alignment — reuse that
  machinery for the exterior rather than inventing a second one.
- `ppgo_exclusion_rho` stays scalar-reach by contract. Do not touch.
- Coordinate axes are SETTLED (Professor, 2026-07-26): exterior(+1)
  additive directional, interior multiplicative, tube (u=sqrt(eta)),
  saddle exterior additive scalar-reach. NO axis changes in this build.

## Out of scope — hard fences

- NO axis/coordinate redesign. NO campaign, NO qd, NO Born rung.
- NO tolerance weakening anywhere — defect 1's acceptance is that the
  measured eps FALLS below the existing bar, not that the bar moves.
- NO changes to the interior tiler's existing cusp alignment or to the
  verified `admits_exterior` caustic-distance invariant (exact-zero
  false admits must still hold after defect 2's fix).

## Acceptance (two-tier)

1. In-build (FAST): (a) exterior tiles are cusp-ray aligned, no
   admitted tile straddles a cusp ray, and the on-ray eps at
   (gamma=0.40, rho=2.183) drops from 2.6e-1 to the off-ray scale
   (~1e-3 or better) — with `EnvelopeReconstructionTestCase` going
   GREEN at its unchanged 0.2 tolerance; (b) tile-level coverage of the
   0.80-0.90 band rises from 0.56 toward the ~0.88 the predicate
   supports, with the exact-zero-false-admit invariant still holding in
   `test_lensing_exterior_admission.py`; (c) the interior false-admit
   case (band (0.45,0.55), theta=0, rho=0.74, exact distance 0.0462)
   is REFUSED, with a reachable-red proving the old cloud test admitted
   it; (d) tube byte-identity; fast tier green.
2. POST-BUILD (driver): calibration pilot on the fixed tiler, then the
   serving census — the first census that can honestly reflect the
   admission repair.
