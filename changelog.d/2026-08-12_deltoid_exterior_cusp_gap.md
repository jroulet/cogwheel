---
date: 2026-08-12
---
### Deltoid exterior cusp gap closed: mid-w ppGO band + MINUS_GHOST saddle exterior window

The exterior cusp neighbourhood (just outside the deltoid cusp tips,
`rho` 1.0–~1.2) is now served quadrature-free on both parities — no
exact-engine serving in the cusp neighbourhood.

- **`_R_PPGO_ERROR_CONST` calibrated 3.0 → 0.10** in
  `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py` so the ppGO fast rung
  fires across the FULL mid-w band (`r_ppgo_min ≈ radius_min ≈ 7.37`),
  closing the astroid mid-w exact-engine flashback and serving saddle
  exterior sources outside the fold band. Measured ppGO accuracy 5e-6 to
  6e-5 vs the exact engine across the opened band, both parities.
- **`ExteriorPolarChart` extended to the saddle exterior cusp window** via
  the `FARFIELD_KERNEL_SUM_MINUS_GHOST` envelope label (ghost gates become
  the admission authority; `_CUSP_EXCLUSION_DISTANCE` reduced for saddle
  near-cusp tiles). No serve-side changes — the MINUS_GHOST mirror already
  existed.

Stale fixtures re-anchored: ppGO-threshold tests, calibration-invoked tests
(ppGO disabled via mock where the uniform path is the probe), and
mpmath-routing tests (arms-first contract). All 367 affected tests pass.
