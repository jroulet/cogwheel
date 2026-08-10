---
date: 2026-08-09
bump: patch
---

## Exterior cusp-exclusion cut (both parities) and ppGO above-ceiling intercept

Document two shipped changes from the exterior_cusp_exclusion_cut and
exterior_followup WP4 builds:

1. **CUSP-EXCLUSION FILTER in FAR-FIELD TILING** (`surrogate_training.py`):
   `_exclude_near_cusp` with `_CUSP_EXCLUSION_DISTANCE = 0.35` (calibrated from
   measured FARFIELD_KERNEL_SUM envelope turn-on) excludes exterior tiles on
   BOTH parities — astroid cusps (`_cusp_source_angles`) AND deltoid/saddle
   cusps (`_deltoid_cusp_source_angles`, D2-folded, both lobes) — checked at
   band edges so a tile is dropped if any band-edge gamma places a corner within
   the exclusion distance. Certified by `test_lensing_exterior_admission.py`.

2. **ABOVE-CEILING ppGO INTERCEPT** in `LensedRelativeBinningLikelihood`
   (`likelihood.py`): `_ppgo_above_ceiling` serves draws where
   `w_max > W_CEILING_SCHWINGER_QD` (=150) AND the narrowest real-image pair is
   resolved (`w_lo * min_delta_tau >= RHO_END`), using fold-corrected ppGO via
   `reconstruct_farfield`. On gate miss falls through to exact engine unchanged.
   Certified by `test_lensing_ppgo_above_ceiling.py`.
