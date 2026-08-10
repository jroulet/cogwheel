---
date: 2026-08-10
---
### Exterior surrogate: 2D (rho, u) fold-carrier

The exterior-polar surrogate's fold-carrier extends from 1D to 2D:
`ExteriorPolarChart` now carries `rho_u_carrier`, an `(n_rho, n_theta_c)`
array of fold-merge-point delays `Re(tau_c(rho, u))` at every spline node.
Training demodulates the envelope by `exp(-1j*w*rho_u_carrier[rho,u])`
before the residual `carrier_rate` demodulation; serve re-modulates in
reverse order, bilinearly interpolating the delay at the query u-coordinate
(after the theta_c → u map). The new axis-schema tag
`'exterior_polar_rho_u_carrier_v2'` (V5) is the current write tag; the V4
`'exterior_polar_rho_log_carrier_v1'` tag stays known for backward
compatibility, and old 1D `rho_carrier` artifacts load by broadcasting to
2D. This removes the residual u-axis phase winding (measured 11.66 rad span,
max dphase/du 48) that the 1D carrier left behind.
