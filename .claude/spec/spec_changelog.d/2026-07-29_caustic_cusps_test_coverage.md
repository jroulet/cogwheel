---
date: 2026-07-29
bump: patch
---

### Certified-by lists gain `test_lensing_caustic_cusps.py`

Doc sync for commit `00bf8ae` (F041 arc-guard fix + the salvaged build-1b
estimator retirement in `surrogate_training.py`). The new test module
`cogwheel/tests/test_lensing_caustic_cusps.py` was missing from every SPEC.md
"Certified by" list; added to both the microlensing-engine row (analytic
`caustic_derivatives`/`caustic_speed`/`caustic_curvature_radius`/
`fold_opening_direction` certification) and the surrogate/training row
(analytic cusp-root, closed-form caustic-inradius, foot-of-normal curvature
value, and fold-orientation-guard certification for the estimators
`surrogate_training.py` retired in favor of the analytic geometry cascade).
No other row-55 (training) prose needed correction: it already described the
guard and cusp-detection behavior generically, without naming the retired
numerical estimators. `COVERAGE_DESIGN.md` needed no change for the same
reason -- its one relevant mention (`_min_curvature_radius`, item C6) is a
behavioral description that still holds under the analytic implementation.
