---
date: 2026-08-08
bump: major
---

### ExteriorPolarChart axis schema -> `exterior_polar_rho_u_v1`: optional cusp-adapted `theta_to_u`

The exterior-polar chart's angular spline axis gains the cusp-adapted
coordinate `u = d**(2/3)` (`d` = angular distance to the NEAR caustic cusp,
`0` or `pi/2` in the D2-folded quadrant) on positive parity, mirroring the
wedge and lobe cusp-adapted axes.

Contract changes, EXTERIOR-POLAR CHARTS ONLY:

- `axis_schema` tag `'exterior_polar_rho_theta_c'` is DROPPED. The ONLY known
  tag is `'exterior_polar_rho_u_v1'` (`_EXTERIOR_POLAR_AXIS_SCHEMA`); an
  absent or unknown tag hard-refuses at load.
- Optional serialized field `theta_to_u` of shape `(2, 2001)` =
  `[theta_c_fine, u_fine]`, built UNIFORM in `u` by the shared
  `_wedge_cusp_axis_map`; `u_grid` is derived as `np.interp` of the training
  `theta_c_grid` through the map, and the spline's 4th axis is `u_grid`
  (concentrating knots near the tile's near cusp). Positive-parity
  (`parity == 1`) production charts always build it; macro-saddle
  (`parity == -1`) exterior charts carry none. An absent `theta_to_u` loads
  as `None` (raw-`theta_c` spline), preserving NPZ round-trip for map-less
  charts.

BREAKING for stored artifacts: every exterior-polar chart written with the
`'exterior_polar_rho_theta_c'` tag fails to load by design, rather than being
served on a mislabelled angular axis. Served `F` is unchanged — serve is
coordinate-agnostic through the stored map.
