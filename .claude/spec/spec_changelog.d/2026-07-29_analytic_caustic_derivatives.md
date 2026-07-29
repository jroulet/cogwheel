---
date: 2026-07-29
bump: minor
---

### `geometry.py` gains analytic caustic derivatives (backward-compatible)

Four new public functions in `cogwheel/lensing/chang_refsdal/geometry.py`,
beside `r_caustic`: `caustic_derivatives` (analytic first and second
theta-derivatives of the closed-form caustic curve, `(y', y'')`),
`caustic_speed` (`|y'|`, vanishes exactly at cusps), `caustic_curvature_radius`
(`|y'|**3 / |y1'y2'' - y2'y1''|`), and `fold_opening_direction` (unit
`D2y[e,e]`, pointing to the two-image side of a fold). All four differentiate
the exact parametric caustic curve directly -- no finite difference, no
`np.gradient`, no sampled-arc stencil.

They add to the module's public surface without replacing anything: the
numerical estimators they will eventually retire (`_min_curvature_radius`,
`_branch_speed_profile`, `_find_cusps`, `_probe_arc_side`, `_cusp_vertex`)
remain in place, pending later builds.
