---
date: 2026-08-08
section: lensing-surrogate
---

### Lobe cusp-adapted coordinate (`u = d**(2/3)`)

`LobeInteriorChart`'s sqrt-edge angular coordinate (`theta_to_s`) is replaced
with the cusp-adapted `u = d**(2/3)` coordinate, `d` the angular distance to
the nearest deltoid cusp vertex — mirroring the `InteriorWedgeChart` v3
pattern (`[[2026-08-07_subdivision-recursion-wedge-v3-r-caustic]]`). The 2/3
exponent is the universal A3 fold-cusp caustic-reach scaling
(`r_deltoid ~ const - c*d**(2/3)`), so `rho_lobe` no longer carries the
`|dtheta|**(1/3)` divergence of the raw `theta_local` axis at a cusp vertex
(and the retired sqrt-edge `s = sqrt(span) - sqrt(theta_max - theta_local)`
axis, designed for A2 folds, did not handle it). The cusp carve-out
(`_LOBE_CUSP_EXCLUSION_DISTANCE`) is retired: the `eta_max` nearest-caustic-
distance test alone excludes near-cusp tiles, and a tile centred at a cusp
vertex clears the eps bar without subdivision.

- `surrogate.py`: `_lobe_cusp_axis_map` (uniform-in-`u` `(2, 2001)` map
  `[theta_fine, u_fine]`, same node count as `_FARFIELD_ARC_MAP_SIZE`);
  schema/field rename `theta_to_s` -> `theta_to_u` / `u_grid`;
  `_LOBE_AXIS_SCHEMA_NEW = 'lobe_caustic_relative_v1'` is the ONLY known lobe
  tag — both OLD tags (`_LOBE_AXIS_SCHEMA_V1` raw-theta, sqrt-edge
  `_LOBE_AXIS_SCHEMA`) hard-refuse at load, and `theta_to_u` is read
  unconditionally so an absent map hard-refuses; `from_lobe_engine` builds
  the cusp-adapted `u` grid via the map.
- `surrogate_training.py`: `_lobe_nearest_cusp` (single authoritative
  nearest-cusp derivation), `_lobe_child_boxes` (angular split at the
  U-MIDPOINT mapped back to `theta_local`, not the raw theta midpoint),
  `_build_lobe_chart` gains `cusp_angle` / `cusp_side`; the
  `_LOBE_CUSP_EXCLUSION_DISTANCE` constant is deleted.

Tests migrated in `test_lensing_surrogate_lobe.py` (u-coordinate round-trips
vs closed-form oracle, `(2, 2001)` shape, bound-shift stability, single-
schema persistence), `test_lensing_lobe_subdivision.py`, and
`test_lensing_wedge_dd_arclength.py`. 149 passed / 16 skipped, 0 failed.

Build history: the build survived a quota-exhaustion crash at the Inspector
stage; the code was salvaged to `b18e6a8`, then a fresh Inspector audit PASS,
Tidier clean (6 unused imports removed), Professor PASS. Professor inference
review PASS (u-coordinate correct, subdivision splits sound, carve-out
retirement correct; one non-blocking note on `_chart_from_npz`'s
unconditional `theta_to_u` access vs `_chart_to_npz`'s conditional write).
