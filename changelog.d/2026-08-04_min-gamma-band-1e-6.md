## 2026-08-04

### Fixed: set min_gamma_band=1e-6 (log-reach spacing makes large floor redundant)

`surrogate_training.stable_gamma_bands` (and its caller
`band_caustic_structure`) now use `min_gamma_band = 1e-6` as the default
bisection floor, replacing the previous `0.005`.

**Why:** the log-reach gamma axis (`1 - gamma`) places nodes well inside
ANY finite-width band regardless of raw-gamma bandwidth; the 0.005 floor
was a legacy guard for the retired uniform-gamma axis.  Setting the floor
to `1e-6` lets bisection proceed to near-float resolution, which is where
the degenerate `gamma = 0` origin produces numerical-noise bands (~1e-10
scale) that the new floor safely discards.

**Effect:** total dropped prior mass ≈ 1.5e-6 (fraction ≈ 1e-6) — Region 10
("dropped slivers") in the coverage-map TODO is now **closed**.

No surrogate geometry, serve path, or chart format is affected; this is a
training-time parameter change only.

Commit: `70affbb`
