---
date: 2026-08-12
section: Backlog
---
# Deltoid (saddle) exterior cusp gap — RESOLVED (commit c8cad0c)

The exterior cusp neighbourhood (just outside the deltoid cusp tips,
`rho` 1.0–~1.2, on and off the lobe axis) is now served QUADRATURE-FREE on
both parities — no exact-engine serving in the cusp neighbourhood, per the
driver mandate.

## Resolution summary (two work packages)

- **WP-1: mid-w ppGO band via `_R_PPGO_ERROR_CONST = 0.10`.** The ppGO fast
  rung's leading-error coefficient was calibrated 3.0 → **0.10**
  (`r_ppgo_min ≈ radius_min ≈ 7.37`), so the rung fires across the FULL
  mid-w band. This closes the astroid mid-w exact-engine flashback
  (`radius_min < R < r_ppgo_min`, where the uniform form did not certify)
  and serves saddle exterior sources outside the fold band. Measured ppGO
  accuracy **5e-6 to 6e-5** vs the exact engine across the opened band,
  both parities. No exact-engine serving remains in the cusp neighbourhood
  (the exterior certificate cannot match 2-image clusters at `R` in
  `(radius_min, 34)`, and ppGO now serves those instead).
- **WP-2: saddle exterior cusp window via the FARFIELD_KERNEL_SUM_MINUS_GHOST
  label.** `ExteriorPolarChart` extends into the saddle exterior cusp
  window: ghost gates become the admission authority and
  `_CUSP_EXCLUSION_DISTANCE` is reduced for saddle near-cusp tiles. The
  serve-side MINUS_GHOST mirror already existed (re-adds the analytically
  subtracted ghost over the chart region with the SAME primitive and
  separation gate), so there were no serve-side changes.

## Related notes

- The surrogate's cusp-exclusion carve-out (`_CUSP_EXCLUSION_DISTANCE = 0.35`)
  was previously the design hole: near-cusp tiles dropped from exterior
  training on both parities, the saddle cusp arm had
  `_SADDLE_CUSP_ARM_COVERAGE = 0.0` (never calibrated), the fold arm
  refused, and the exact engine refused (`SchwingerCertificationError`).
  WP-1 + WP-2 close that hole without touching the surrogate tiler.
- Stale fixtures re-anchored: ppGO-threshold tests, calibration-invoked
  tests (ppGO disabled via mock where the uniform path is the probe), and
  mpmath-routing tests (arms-first contract). All 367 affected tests pass.

## Acceptance

Deltoid exterior sources just outside the cusp tip (rho 1.0–~1.2) are served
by a fast path (ppGO mid-w band, or the MINUS_GHOST exterior surrogate)
across the w band; no live quadrature in the fast path; refusal-conservative.
