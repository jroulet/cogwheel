# Build Brief: Cusp arm boundary — sweep actual serve/refuse boundary

## Mission

The Pearcey table is shipped (cogwheel/data/pearcey_table.npz, 420KB).
The analytic R-gate reach measurement (scripts/measure_cusp_arm_reach.py)
found minimum reach = 0.328 rad, but boundary verification showed that
`cusp_amplification` REFUSES at that boundary — the calibration gate
also binds, not just R >= R_min.

We need to find `cusp_amplification`'s ACTUAL accept/refuse boundary by
sweeping it directly, then set `_CUSP_ARM_COVERAGE` to the measured
minimum.

## Implementation

Write `scripts/measure_cusp_arm_actual_boundary.py` that:

1. Load the Pearcey table via `use_pearcey_table('cogwheel/data/pearcey_table.npz')`.
2. For representative configs:
   - gamma_values = [0.1, 0.2, 0.3, 0.5] (positive) + [1.2, 1.5] (saddle)
   - w_values = [10, 20, 40]
3. At each (gamma, w):
   - Find the cusp angle (theta=0 for positive branch=1 astroid cusp)
   - Binary-search delta_theta outward from the cusp:
     - At each delta_theta, compute source on the caustic via
       `geometry.critical_point(gamma, 0.0, 0.0, 0.0, branch=1)` at the
       cusp, then offset the SOURCE by moving along the critical curve
     - Call `cusp_amplification(np.array([w]), source_2d, gamma)`
     - Record whether it returns None (refuses) or a value (serves)
   - Find the SMALLEST delta_theta where it transitions from refuse → serve
     (the arm serves OUTSIDE a minimum radius, refuses INSIDE)
4. Report the minimum delta_theta across all (gamma, w) — this is the
   actual arm reach.
5. Set `_CUSP_ARM_COVERAGE` in surrogate.py to this value.

## Key API notes

- `cusp_amplification(w, source, gamma)` where:
  - w: 1D array of frequencies
  - source: 2D array shape (2,) — source position in eigenframe
  - gamma: float
  - Returns: complex array or None (refuses)
- The source must be a 2D point (y1, y2), NOT an angle.
- To get sources near a cusp: use `geometry.critical_point` to get the
  IMAGE position on the critical curve, then the corresponding SOURCE
  is obtained from the lens equation (image - deflection). OR: just
  sample source positions radially outward from the cusp vertex source
  position in the soft direction.

## Acceptance

- `_CUSP_ARM_COVERAGE > 0` in surrogate.py (arm enabled).
- At the coverage boundary, `cusp_amplification` returns non-None.
- Census shows cusp-window draws partially served.

## Constraints

- Fast (no engine calls needed — just cusp_amplification + table lookups).
- Follow AGENTS.md and the spec/TODO workflow.
