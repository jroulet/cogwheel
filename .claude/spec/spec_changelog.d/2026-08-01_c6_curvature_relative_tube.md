---
date: 2026-08-01
bump: patch
---

C6: tube shell is now curvature-relative. `TrainingConfig.eta_max`/`eta_floor`
renamed to `f_max`/`f_floor`; per-arc `eta_max = f_max * R_c` computed in
`_train_band_charts` and threaded to `_build_tube_chart`,
`_tube_heldout_samples`, `_interior_admission`, and `_saddle_lobe_admissions`.
The foot-of-normal skip guard is replaced by `assert f_max < 0.5` — vacuous by
construction so no chart is ever skipped for curvature. SPEC.md updated to
reflect the skip is replaced by an assertion.
