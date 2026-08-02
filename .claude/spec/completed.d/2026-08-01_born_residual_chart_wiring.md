---
date: 2026-08-01
section: Lensing
---
### BornResidualChart wiring infrastructure (Build C11)

Introduced `cogwheel/lensing/born_residual_chart.py` — frozen 3-D
interpolation dataclass `BornResidualChart` for the Born-annulus residual
`R(w; gamma, rho) = F_exact - F_carrier`.  Wired the fact-4 slot in
`likelihood._surrogate_coefficients`: when a `BornResidualChart` is attached
to the likelihood object, the slot reconstructs `F_carrier + R` and returns
surrogate coefficients; when the chart is `None` (default), annulus draws
fall through to the exact engine unchanged.  Companion test suite
(`test_lensing_born_residual_wiring.py`, 34 tests) certifies the wiring,
kappa/beta guard precedence, and axis-aligned coverage checks.
