---
date: 2026-08-20
---

### Low-w diffractive serve: deep interior now calibrated, not clipped

The truncation-certificate fit (`w_low_fit`) that admits the positive-
parity low-w analytic rung was re-calibrated to cover the deep interior:
the calibration grid now reaches reduced radii down to `r ~ 0.1`, so the
fit is trained where `gamma' * s * w / 2` is genuinely small. The deep
interior is now served conservatively by the calibrated, de-rated fit at
its engine-honest ceiling (a smooth monotone function of `rho`, ~4-41),
rather than by the `min(w_fit, ceiling)` clip, which was quietly
re-serving the interior at the 60 cap where the series is not honest
(measured over-serve up to ~2.9x before this build). The de-rate is now
the sole conservativeness margin; the ceiling clip remains only as a
hard oracle-domain cap. Coefficients baked at this stage are provisional
smoke-scale values pending a full-scale driver re-bake.
