---
date: 2026-08-20
---

### Low-w diffractive band: near-fold shell and wall band now chart-served

The positive-parity low-w diffractive serve gained a trained 4-D residual
chart (`LowWDiffractiveChart`, package data `low_w_diffractive_chart.npz`)
that serves the two regimes the truncation-certificate fit cannot: the
near-fold shell (where `w_low_fit` declines) and the wall band (where its
order-16 series collapses). The chart interpolates the smooth reduced-frame
residual `r_pure = f_pure / (sqrt(mu_pure) * prefactor_c(w))` over
caustic-relative coordinates and is re-modulated at serve time; the exact
Schwinger engine stays an offline training oracle only, never a serve-time
call. A single scalar de-rate is the sole conservativeness margin, and cells
the training oracle flags as unable to meet the 1e-4 certification bar are
declined per-cell — a covered draw there falls through to the exact engine
rather than being amplitude-scaled. The serve-route census mirrors the new
route as its 12th `SERVE_ROUTE` (`low_w_diffractive_chart`).
