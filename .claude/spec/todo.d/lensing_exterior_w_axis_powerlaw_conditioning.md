---
section: Backlog
depends_on: [2026-08-09_exterior_cusp_exclusion_cut]
---

- **Exterior envelope w-axis power-law conditioning (log-scale fit)**
  `[→ spec]` — identified 2026-08-09 by Professor + Coder investigation.

  The exterior `E_ff` label is SMOOTH and correctly non-oscillatory (full
  kernel subtraction works; the earlier "beat" claim was a measurement
  artifact). But `|E(w)| ~ w^(-0.60)` (R²=0.996): a clean power-law decay
  spanning ~1000× over the w-band. A cubic spline with ~7 w-nodes
  (4/decade) is exact at nodes (1e-17) but 12%–1200% off between them,
  because the steep power-law curvature can't be tracked at low node
  counts. The eps bar (1e-3 normalized by max|F|) is breached.

  The fix must be a COORDINATE/SCALE transform, not added resolution (per
  the engineering principle: spline smooth things, don't out-resolve
  steep ones). The magnitude follows a clean power law, suggesting a
  log-scale or power-law-rescaling coordinate on the w-axis. BUT the
  envelope is complex (real/imag with rotating phase), and the spatial
  axes (rho, theta_c, gamma) interact — a log|F| ordinate alone is not
  self-consistent. The build must DESIGN the full representation: how to
  fit a complex envelope with ~1000× dynamic range at low node counts,
  including phase handling and whether the spatial axes need matching
  transforms, and keep the serve path (reconstruct_farfield) consistent.

  ACCEPTANCE: the exterior surrogate clears the 1e-3 eps bar at the probe's
  4×4×4 (or modestly higher) node count WITHOUT resolving the decay by
  density alone; the transform round-trips to the exact F at machine
  precision; the serve path and reconstruction are consistent.
