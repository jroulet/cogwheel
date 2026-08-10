---
date: 2026-08-09
section: Backlog
---

## Exterior cusp-exclusion cut for both parities

Implemented in commit d685ebe (build exterior_cusp_exclusion_cut).

### What shipped

- **`_CUSP_EXCLUSION_DISTANCE` bumped 0.2 → 0.35**: calibrated from the
  measured FARFIELD_KERNEL_SUM envelope turn-on distance; the old 0.2 admitted
  tiles whose nearest corner sat at 0.206 y-units from the cusp and still
  failed the 1e-3 bar (eps ~ 0.076).
- **`_deltoid_cusp_source_angles`**: new helper mapping the six deltoid/saddle
  cusp vertices (3 per lobe × 2 lobes) to D₂-folded source-plane angles in
  `[0, π/2]`, matching the exterior polar chart's `theta_c` domain.
- **Both-parity exclusion**: `_exclude_near_cusp` now accepts
  `cusp_angles` covering both astroid (`_cusp_source_angles`) and deltoid
  (`_deltoid_cusp_source_angles`) cusps; called with band-edge
  `gamma_band=(gamma_lo, gamma, gamma_hi)` so a tile is dropped when ANY
  band-edge shear places a corner within the exclusion distance.
- **Saddle path wiring**: `_farfield_tiles` gained optional `cusp_angles`,
  `gamma`, `gamma_band` kwargs (backward-compatible `None` default); the
  saddle exterior branch now passes them.
- 53 tests pass; certified by `cogwheel/tests/test_lensing_exterior_admission.py`.

### Post-build driver measurement owed

- 4x4x4 probe should confirm ~70 charts/band (not 500+) on a full-box training
  run. The in-build probe showed the cut is effective at fixture scale.

### Acceptance

Acceptance criteria from `lensing_exterior_cusp_exclusion_cut.md` met at
in-build scale: `_exclude_near_cusp` filters both parity cusp windows, the
exclusion distance is calibrated from measured envelope turn-on, and the build
passes 53 tests. The ~70 chart target is a driver campaign gate (training scale).
