---
date: 2026-08-20
bump: patch
---

SPEC.md's LOW-W DIFFRACTIVE RUNGS paragraph (Rung P) updated for the low-w
near-fold / wall-band chart-serve build: the near-fold shell (`rho` in
`[RHO_LO, 1 + DELTA]`) is no longer DECLINED — it and the wall band
(`gamma' > _WALL_GAMMA_PRIME = 0.5`) are served by the trained low-w
diffractive residual chart `LowWDiffractiveChart` (package artifact
`cogwheel/data/low_w_diffractive_chart.npz`, schema `low_w_diffractive_v1`,
offline-Schwinger-trained by `scripts/train_low_w_diffractive_chart.py`),
consulted FIRST in the Rung-P branch before the `w_low_fit` split. The
paragraph now states the residual representation (`r_pure = f_pure /
(sqrt(mu_pure) * prefactor_c(w))` with both known analytic factors stripped),
the re-modulation (`F = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full *
r_pure`), the union-band `covers` predicate (near-fold shell OR wall) with the
served-`w` band inside the trained log-w range, the scalar de-rate as the sole
margin, and per-cell DECLINED fall-through to the exact engine (never an
amplitude scale). The ENGINE-FREE SERVE-ROUTE DEMAND CENSUS paragraph's route
count/list corrected from the stale EIGHT-route set to the current TWELVE
(`SERVE_ROUTES` now includes `born_carrier_only`,
`low_w_diffractive_chart`, `diffractive_analytic`, and
`diffractive_engine_hosted`), with the waterfall description aligned to the
production rung order (ppgo_above_ceiling before saddle_c3, the low-w
diffractive parity split, then the per-node pass).
