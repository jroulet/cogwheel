---
section: Backlog
depends_on: [2026-08-09_exterior_cusp_exclusion_cut]
---

- **Exterior envelope w-axis power-law conditioning (log-scale fit)**
  `[→ spec]` — identified 2026-08-09 by Professor + Coder investigation.

  **W-AXIS DONE (2026-08-10, build exterior_w_axis_powerlaw_conditioning, commit f4652e7):**
  Carrier-phase demodulation (`carrier_rate = k_chart`) implemented in
  `ExteriorPolarChart`. Per-node unwrapped-phase slope is medianed to
  `k_chart`; the stored envelope is demodulated by `exp(-1j*k_chart*w)`
  before spline fitting (`from_values`, single canonical site) and
  re-modulated at serve. Axis schema bumped to
  `'exterior_polar_carrier_demod_v2'` (old tag hard-refuses). W-axis
  off-grid eps now ~1e-4 (node eps 1e-17), well below the 1e-3 bar.

  **REMAINING — SPATIAL AXES:** the carrier fix exposes the spatial-axis
  problem: `rho` spans ~3 decades toward rho=1 and `theta_c` also need
  coordinate conditioning; off-grid rho eps ~0.04 at 4 nodes/decade.
  The full acceptance bar is NOT yet cleared. The spatial-axis conditioning
  is a follow-on build (not yet in a TODO fragment).

  The fix must be a COORDINATE/SCALE transform on the spatial axes too,
  not added resolution. The build must keep the serve path
  (`reconstruct_farfield`) consistent.

  ACCEPTANCE: the exterior surrogate clears the 1e-3 eps bar at the probe's
  4×4×4 (or modestly higher) node count WITHOUT resolving the decay by
  density alone; the transform round-trips to the exact F at machine
  precision; the serve path and reconstruction are consistent.
