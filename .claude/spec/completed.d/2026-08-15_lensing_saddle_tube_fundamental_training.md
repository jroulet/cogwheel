---
date: 2026-08-15
section: Lensing serving
---

Closed `todo.d/lensing_saddle_tube_fundamental_training.md`. The saddle
branch of `_tube_training_arcs` now derives a D2-orbit partition from the
fold law (midpoint-angle clustering across the four gauge images,
tolerance `max(1e-3, 0.25*min_width)`) instead of slicing
`arcs[:max_tube_arcs]`; the `max_tube_arcs` field is removed from
`TrainingConfig` entirely, matching the TODO's "MUST land before the
training campaign" directive. Detected-vs-trained count split mirrors
the astroid test's existing pattern (typically 6 -> 3 for the saddle).

F081's starvation fix rides along in the same build: `_train_band_charts`
now computes `min_eta_max` alongside `max_eta_max` and feeds
`min_eta_max` to `saddle_lobe_admissions` and the deltoid far-field
`physical_exclusion_radius`, replacing the isotropic band-wide `max()`
F081 identified as the starvation mechanism. Both defects F081 named
(config: heterogeneous per-arc `r_min` from training all 6 arcs; wiring:
`max()` applied isotropically) are addressed by this build — see
`.claude/spec/FINDINGS.md` F081's RESOLVED marker.

Mirrored into `cogwheel/lensing/tiling_census.py`
(`_REQUIRED_CONFIG_FIELDS`, `_build_band_ctx`), `scripts/census_dry_run.py`,
and `scripts/train_surrogate_production.py`. `SPEC.md`'s TUBE D2
GAUGE-IMAGE FOLD paragraph updated to describe both-parity orbit
training and the F081 fix; spec bumped minor
(`spec_changelog.d/2026-08-15_saddle_tube_fundamental_training.md`).
