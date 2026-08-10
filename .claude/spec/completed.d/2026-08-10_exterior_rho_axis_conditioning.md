---
date: 2026-08-10
section: Backlog
---
## Exterior envelope rho-axis conditioning (log(rho-1) coordinate)

Implemented in build `exterior_rho_axis_conditioning`, commit f6b8b05.

- `ExteriorPolarChart` gains `rho_log_axis` flag: when True, rho axis is
  reparameterized to `ur = log(rho-1)`, linearizing the ~4.5-decade magnitude
  growth toward rho=1. Enabled at training (`_build_farfield_chart`, both
  parities). Composes with w-carrier demodulation from the preceding build.
- Schema bumped from `exterior_polar_carrier_demod_v2` to
  `exterior_polar_rho_log_v3` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V3`); old tag
  hard-refuses at load.
- `rho_log_axis=False` default is byte-identical to prior HEAD.
- Tree-gate regression fixed: 7 test references to the renamed constant
  `_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER` updated to `_EXTERIOR_POLAR_AXIS_SCHEMA_V3`.
- 99+215 tests pass (test_lensing_exterior_carrier + test_lensing_surrogate).

ACCEPTANCE: exterior surrogate clears the 1e-3 eps bar at the probe's node
count with BOTH the w-carrier and rho-coordinate fixes; round-trip to machine
precision; serve path consistent. Bulk training sweep is a driver post-build step.
