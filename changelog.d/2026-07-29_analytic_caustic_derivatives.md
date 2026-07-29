---
date: 2026-07-29
---

### Added: analytic caustic-curve derivatives in the microlensing geometry module

`cogwheel/lensing/chang_refsdal/geometry.py` gains four public functions,
beside `r_caustic`: `caustic_derivatives` (analytic first and second
theta-derivatives of the closed-form Chang-Refsdal caustic, `(y', y'')`),
`caustic_speed` (`|y'|`, an exact root at cusps), `caustic_curvature_radius`
(`|y'|**3 / |y1'y2'' - y2'y1''|`), and `fold_opening_direction` (the unit
vector pointing to a fold's two-image side).

All four differentiate the exact parametric caustic curve directly, with no
finite difference, no `np.gradient`, and no sampled-arc stencil anywhere in
the new code. They add to the existing public surface; nothing is removed or
retired by this change.
