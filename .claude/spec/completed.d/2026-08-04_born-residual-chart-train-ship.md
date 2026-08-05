---
date: 2026-08-04
section: Backlog
---

### Born carrier + band-split residual chart — train step complete

The final TRAIN_TIER step of `todo.d/lensing_born_b1_derivation.md` is done.

Commit `849e580` trained and shipped `cogwheel/data/born_residual_chart.npz`
(approx 8 KB, package data): a 3-D tensor-product cubic spline of the Born
residual `R(w; gamma, rho) = F_exact_demod(w) - F_carrier_demod(w)` over
a 7 gamma x 5 rho x 10 log-w sparse grid, min-relative delay frame.

The fact-4 slot in `likelihood._surrogate_coefficients` (landed C11,
2026-08-01) attaches the chart at construction time; when attached, exterior
draws (`rho > 1.0` within the grid) are served as `carrier + residual`
without running the exact engine.  When `None` (default), the fall-through
to the exact engine is the correct behavior and remains so.

SPEC.md and DATA_CONTRACTS.yaml updated in commit `8e668a2`.
Saddle branch: the same residual chart covers both parities; see
`todo.d/lensing_saddle_born.md` (step 0's fence issue still open).
