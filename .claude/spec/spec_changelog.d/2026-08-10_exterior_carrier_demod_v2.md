---
date: 2026-08-10
bump: patch
---

## Exterior carrier demodulation: axis schema bump to exterior_polar_carrier_demod_v2

Document the shipped carrier-demodulation mechanism in `ExteriorPolarChart`
(commit f4652e7):

- Axis schema bumped from `'exterior_polar_rho_u_v1'` to
  `'exterior_polar_carrier_demod_v2'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER`);
  both retired tags (`'exterior_polar_rho_theta_c'` and
  `'exterior_polar_rho_u_v1'`) hard-refuse at load.
- New `carrier_rate` field (float, default 0.0): per-node unwrapped-phase
  slope medianed to `k_chart`; stored envelope demodulated by
  `exp(-1j*k_chart*w)` before spline fitting (single canonical site:
  `from_values`) and re-modulated at serve, making the complex exterior
  envelope splineable despite ~1000x dynamic range (`|E(w)| ~ w^(-0.60)`).
- Updated in SPEC.md Key abstractions and DATA_CONTRACTS.yaml description.
