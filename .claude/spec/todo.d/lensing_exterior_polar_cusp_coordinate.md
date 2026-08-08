---
section: Backlog
depends_on: [2026-08-08_lobe-cusp-adapted-coordinate]
---

- **Exterior polar chart needs `u = d^(2/3)` angular coordinate**
  `[→ spec]` — identified 2026-08-08 by Professor.

  `ExteriorPolarChart` uses ``(rho, theta_c)``. Near cusp angles
  (``theta_c → 0, π/2``) the caustic radius ``r_caustic ~ const -
  c d^(2/3)`` makes the envelope ``E(rho, theta_c, w)`` vary as
  ``d^{-1/3}`` in theta_c — a cubic spline on uniform theta_c nodes
  cannot fit it.  Result: aggressive subdivision, 500+ charts for a
  4×4×4 probe.

  **Fix**: ``theta_c → u = d^(2/3)`` where ``d`` is angular distance
  to the nearer cusp in the D₂-folded quadrant (``d = theta_c`` or
  ``d = π/2 - theta_c``).  ``dE/du`` is finite — the exponent cancels
  the ``d^(2/3)`` in ``r_caustic`` exactly.  Same pattern as
  `InteriorWedgeChart`'s ``u`` coordinate (wedge v3, F064), using the
  same ``_wedge_cusp_axis_map`` helper and ``theta_to_u`` serialization.
  ``rho`` stays as-is (``drho/d|y| = 1``, well-behaved).

  Saddle exterior (``gamma > 1``) uses a scalar ``rho`` without
  directional ``r_caustic`` — no angular cusp singularity, no change
  needed.

  ACCEPTANCE: a 4×4×4 probe produces ~70 charts (not 500); a tile at
  ``u=0`` (cusp vertex) clears the 1e-3 bar; the serving path maps
  ``theta_c → u`` via ``np.interp`` before spline contraction.
