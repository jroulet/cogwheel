---
bump: patch
date: 2026-08-10
---
## Exterior rho-axis conditioning: axis schema bump to exterior_polar_rho_log_v3

- `ExteriorPolarChart` `axis_schema` bumped from `'exterior_polar_carrier_demod_v2'`
  to `'exterior_polar_rho_log_v3'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V3`); retired
  `'exterior_polar_rho_theta_c'`, `'exterior_polar_rho_u_v1'`, and
  `'exterior_polar_carrier_demod_v2'` all hard-refuse at load.
- New field `rho_log_axis` (bool, default False) serialized per-chart in the npz
  meta dict: when True, the rho axis in the stored spline is `ur = log(rho-1)`;
  inverted transparently at serve. Default False preserves byte-identity with
  pre-v3 builds (but stale v2 artifacts still hard-refuse on the schema tag).
