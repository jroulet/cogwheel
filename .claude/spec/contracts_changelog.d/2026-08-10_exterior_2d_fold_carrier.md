---
date: 2026-08-10
bump: minor
---
Updated the `lens_amplification_surrogate` contract's ExteriorPolarChart
description: `rho_carrier` (1-D `(n_rho,)`) replaced by the 2-D
`rho_u_carrier` (`(n_rho, n_theta_c)`, `Re(tau_c(rho, u))` at each spline
node); `axis_schema` now lists TWO known tags — V4
`'exterior_polar_rho_log_carrier_v1'` (retained for backward compatibility)
and V5 `'exterior_polar_rho_u_carrier_v2'` (current write tag) — with old 1-D
`rho_carrier` artifacts loading via broadcast to 2-D.
