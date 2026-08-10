---
bump: patch
date: 2026-08-10
---
## Exterior rho-axis conditioning: axis schema bump to exterior_polar_rho_log_v3

- `ExteriorPolarChart` axis schema bumped from `'exterior_polar_carrier_demod_v2'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER`) to `'exterior_polar_rho_log_v3'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA_V3`); all three retired tags now hard-refuse at load.
- New serialized field `rho_log_axis` (bool, default False): when True, the rho
  spline axis is reparameterized to `ur = log(rho-1)`, linearizing the ~4.5-decade
  envelope magnitude growth toward `rho=1`. Applied at training (`from_values`);
  inverted transparently at serve.
- Key abstractions section updated: schema tag, constant, retired-tag list,
  and `rho_log_axis` field description.
