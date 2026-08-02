---
date: 2026-08-01
section: Backlog
---

C6 (tube shell curvature-relative) from `lensing_caustic_relative_coordinates`.
`TrainingConfig.eta_max`/`eta_floor` → `f_max`/`f_floor`; per-arc absolute
`eta_max = f_max * R_c` computed in `_train_band_charts` and threaded to four
private training functions. Foot-of-normal skip guard deleted; replaced by
`assert f_max < 0.5`. No chart skipped for curvature. SPEC.md updated.
Coverage-map region 3 foot-of-normal cause marked closed.
