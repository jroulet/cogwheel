---
date: 2026-08-10
bump: patch
---

## Exterior polar axis schema updated to carrier_demod_v2

Update `lens_amplification_surrogate` contract description for
`ExteriorPolarChart` records (commit f4652e7):

- `axis_schema` changed from `'exterior_polar_rho_u_v1'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA`) to `'exterior_polar_carrier_demod_v2'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER`); both retired tags hard-refuse
  at load.
- New `carrier_rate` field (float, default 0.0) documented: residual
  carrier-phase rate `k_chart`; when nonzero, envelope is demodulated by
  `exp(-1j*k_chart*w)` before spline fitting and re-modulated at serve.
