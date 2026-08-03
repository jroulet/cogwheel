---
bump: patch
---

Update training section to reflect `min_gamma_band = 1e-6` default: bisection
now continues to near-float resolution with negligible dropped prior mass
(~1e-6 fraction). Update `test_lensing_min_gamma_band.py` cert blurb to
describe threshold-discriminant invariants at explicit non-zero widths (the
production default is 1e-6). Close region 10 in the coverage-map TODO fragment.
