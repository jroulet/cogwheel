---
date: 2026-08-20
bump: minor
---

### Register the `low_w_diffractive_chart` data product

New artifact entry: the trained low-w diffractive residual chart shipped as
package data at `cogwheel/data/low_w_diffractive_chart.npz` (schema
`low_w_diffractive_v1`), produced by `scripts/train_low_w_diffractive_chart.py`
and consumed by `LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve`
and the serve-route census's `classify_draw`. The entry records the 4-D
reduced-coordinate residual representation (`r_pure = f_pure /
(sqrt(mu_pure) * prefactor_c(w))` with both known analytic factors stripped),
the 8 npz fields including the scalar `derate` and the per-cell
`declined_mask`, the content-hash provenance convention, and the chart's
first-consulted union-band serve (near-fold shell OR wall band) with per-cell
decline fall-through to the exact engine.
