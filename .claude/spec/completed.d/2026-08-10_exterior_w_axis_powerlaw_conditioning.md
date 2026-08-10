---
date: 2026-08-10
section: Backlog
---
## Exterior envelope w-axis power-law conditioning (carrier demodulation + rho-axis)

W-axis done in build `exterior_w_axis_powerlaw_conditioning`, commit f4652e7.
Spatial-axis (rho) conditioning done in build `exterior_rho_axis_conditioning`,
commit f6b8b05. Both parts required for the acceptance bar.

- W-axis: carrier-phase demodulation (`carrier_rate = k_chart`) implemented in
  `ExteriorPolarChart`. Per-node unwrapped-phase slope medianed to `k_chart`;
  stored envelope demodulated by `exp(-1j*k_chart*w)` before spline fitting and
  re-modulated at serve. W-axis off-grid eps now ~1e-4 (node eps 1e-17).
- Rho-axis: `rho_log_axis=True` reparameterizes rho to `log(rho-1)`, linearizing
  the ~4.5-decade magnitude growth toward rho=1. Off-grid rho eps improved from
  ~0.04 to below the 1e-3 bar.
- Both fixes compose; final schema `exterior_polar_rho_log_v3`.

ACCEPTANCE: exterior surrogate clears the 1e-3 eps bar at the probe's node count
with BOTH the w-carrier and rho-coordinate fixes. Bulk training sweep is a driver
post-build step.
