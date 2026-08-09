---
date: 2026-08-08
bump: minor
---

### Exterior-polar charts use the cusp-adapted `u = d**(2/3)` angular coordinate

The `ExteriorPolarChart` coordinate contract (Key abstractions — "Far-field
surrogate coordinate contract") is updated to the shipping cusp-adapted axis.
Positive-parity (`parity == 1`) exterior charts carry an optional
`theta_to_u` map: the spline's 4th axis is then `u = d**(2/3)`, `d` the
angular distance to the NEAR caustic cusp (`0` or `pi/2` in the D2-folded
quadrant) — the same gamma-universal cusp-reach scaling the wedge and lobe
charts use, absorbing the `d**(-1/3)` near-cusp divergence in `dE/dtheta_c`.
Macro-saddle (`parity == -1`) exterior charts interpolate on raw `theta_c`
(no map).

The axis-schema tag is bumped from `'exterior_polar_rho_theta_c'` to
`'exterior_polar_rho_u_v1'` (`_EXTERIOR_POLAR_AXIS_SCHEMA`); the retired tag
is dropped from the known set and hard-refuses at load. "No arc-length map is
needed" remains true — the optional map is cusp-adapted, not arc-length.
