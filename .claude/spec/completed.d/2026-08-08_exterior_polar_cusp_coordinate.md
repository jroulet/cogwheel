---
date: 2026-08-08
section: lensing-surrogate
---

### Exterior polar chart cusp-adapted `u = d**(2/3)` angular coordinate

`ExteriorPolarChart`'s raw `theta_c` angular spline axis is replaced (on
positive parity) with the cusp-adapted `u = d**(2/3)` coordinate, `d` the
angular distance to the NEAR caustic cusp (`0` or `pi/2` in the D2-folded
quadrant) — the same gamma-universal caustic-reach scaling
(`r_caustic ~ const - c*d**(2/3)`) the wedge v3 and lobe charts use, so the
spline absorbs the `d**(-1/3)` divergence in `dE/dtheta_c` near cusp angles
that forced aggressive subdivision. Macro-saddle (parity == -1) exterior
charts are unchanged (scalar-reach `rho`, raw `theta_c`).

- `surrogate.py`: optional `theta_to_u` / `u_grid` on `ExteriorPolarChart`
  (from_values / from_engine / _assemble); `_evaluate_chart` maps
  `theta_c -> u` via `np.interp` before spline contraction; axis schema
  bumped `'exterior_polar_rho_theta_c'` -> `'exterior_polar_rho_u_v1'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA`, the ONLY known tag — the retired tag
  hard-refuses at load); `_chart_to_npz` conditionally writes the map and
  `_chart_from_npz` reads it via `data.get` (None fallback), preserving NPZ
  round-trip for map-less charts.
- `surrogate_training.py`: `_build_farfield_chart` trains parity == 1 tiles
  with the cusp-adapted map via the shared `_wedge_cusp_axis_map`
  (waist-derived `origin`); parity == -1 passes None (raw-theta fallback).

Tests in `test_lensing_exterior_polar_fold.py`,
`test_lensing_exterior_windows.py`, `test_lensing_farfield_envelope.py`,
`test_lensing_surrogate.py`, `test_lensing_surrogate_census.py`,
`test_lensing_surrogate_lobe.py`, `test_lensing_surrogate_training.py`, and
`test_lensing_wedge_dd_arclength.py` (cusp-adapted axis construction,
serve-path self-falsification, NPZ round-trip with/without the map, census
coverage). SPEC.md and DATA_CONTRACTS.yaml synced by Librarian (schema tag
and coordinate description; changelog fragments
`2026-08-08_exterior_polar_cusp_axis`).

Driver post-build verification still open: the training-scale acceptance —
a 4x4x4 probe producing ~70 charts rather than 500, and a cusp-vertex tile
clearing the far-field eps bar — is a bulk-training sweep, not an in-build
test.
