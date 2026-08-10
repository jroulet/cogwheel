---
date: 2026-08-10
bump: minor
---
Updated the exterior-polar surrogate coordinate contract (Key abstractions):
the fold-carrier is now the 2-D `rho_u_carrier` (`(n_rho, n_theta_c)`,
`Re(tau_c(rho, u))` at each spline node) replacing the 1-D `rho_carrier`;
listed BOTH known axis-schema tags — V4 `'exterior_polar_rho_log_carrier_v1'`
(`_EXTERIOR_POLAR_AXIS_SCHEMA_V4`, retained for backward compatibility) and
V5 `'exterior_polar_rho_u_carrier_v2'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V5`, the
current write tag) — and described demodulation by
`exp(-1j*w*rho_u_carrier[rho,u])` with serve re-modulation at the
u-coordinate. Also updated the FOLD-CARRIER DEMODULATION sentence in the
far-field tiling narrative (`_compute_rho_carrier` → `_compute_rho_u_carrier`).
