---
date: 2026-08-10
section: lensing-surrogate
---
## Exterior 2D (rho, u) fold-carrier

Implemented in build `exterior_2d_fold_carrier` (code in working tree,
Inspector PASS). Extends the 1-D rho-carrier of b061103 to the 2-D
`(rho, u)` fold-carrier.

- `_compute_rho_carrier` replaced by `_compute_rho_u_carrier` in
  `surrogate.py`: the fold-carrier is now a 2-D `(n_rho, n_theta_c)` array of
  `Re(tau_c(rho, u))` — the fold-merge-point delay — at EVERY spline node,
  not a per-rho median. For each `(rho, theta_c)` node it probes
  `geometry.ghost_kernel` and takes the median `Re(tau_c)` over `gamma`
  (`theta_c_grid[j]` is the u-axis partner of `u_grid[j]`, index-paired, no
  inverse interpolation needed). NaN nodes (no ghost) are filled
  conservatively: linear along u, then rho; zero-order hold at boundaries;
  all-NaN → None.
- `ExteriorPolarChart` field `rho_carrier` → `rho_u_carrier` (np.ndarray or
  None, default None, shape `(n_rho, n_theta_c)`): `from_values` demodulates
  the envelope by `exp(-1j*w*rho_u_carrier[rho, u])` BEFORE the residual
  `carrier_rate` demodulation; serve re-modulates in reverse order, the
  `rho_u_carrier` delay bilinearly interpolated at the query u-coordinate
  (after the theta_c → u map, never raw theta_c).
- Schema: NEW V5 `'exterior_polar_rho_u_carrier_v2'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA_V5`) is the current write tag; V4
  `'exterior_polar_rho_log_carrier_v1'` stays in the known set for backward
  compatibility. Old 1-D `rho_carrier` artifacts load by broadcasting to 2-D
  (backward-compatible); the retired raw-theta/v1/v2/v3 tags still hard-refuse.
- Motivation (probe): the 1-D carrier left the u-axis winding — measured
  11.66 rad phase span in u, max dphase/du 48 (82 on raw theta_c); the 2-D
  carrier flattens the per-rho phase span in u to <= 1.63 rad, splineable at
  4 nodes/axis.

ACCEPTANCE (driver post-build verification, not in-build): the full-box
exterior probe produces ~70 charts with all held-out eps under the 1e-3 bar;
the u-axis off-grid eps drops below the bar; round-trip to machine precision;
serve re-modulation at the interpolated u verified.
