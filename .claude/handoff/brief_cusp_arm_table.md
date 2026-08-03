# Build Brief: Ship Pearcey table and enable cusp arm coverage

## Mission

The Pearcey cusp arm machinery is fully implemented (`_pearcey_cusp.py`,
`_pearcey_table.py`, `PearceyTable`) but disabled: `_CUSP_ARM_COVERAGE = 0.0`
in `surrogate.py`. No `pearcey_table.npz` is shipped in `cogwheel/data/`.
This means cusp neighbourhoods (coverage region 4) fall through to exact
quadrature on every serve call.

Enable it:
1. Measure the arm's angular reach (how far from the cusp vertex the
   Pearcey uniform approximation stays within the serving eps bar).
2. Generate and ship `pearcey_table.npz`.
3. Set `_CUSP_ARM_COVERAGE` to the measured reach.

## Context

- `_pearcey_table.py` already defines `PearceyTable` (frozen dataclass with
  `from_grid` class method, NPZ serialization, `__call__` interpolation).
- `_pearcey_cusp.py` has `use_pearcey_table(path)` to load a table and
  `pearcey_cusp_term(...)` that uses it when available.
- `surrogate.py` line 2832: `residual = max(0.0, delta_theta - _CUSP_ARM_COVERAGE)`
  — this is where the coverage constant gates how much cusp window is served
  vs falls through.
- The F016 envelope bar is the acceptance criterion for served values.

## Implementation

1. Write `scripts/build_pearcey_table.py` that:
   - Generates a PearceyTable on a grid covering the required (x, y) range
   - The range should cover the physical Pearcey controls encountered at
     representative (gamma, theta_cusp, w) values
   - Saves to `cogwheel/data/pearcey_table.npz`

2. Write `scripts/measure_cusp_arm_reach.py` that:
   - At representative cusps (gamma=0.1, 0.3, 0.5 positive; 1.2, 1.5 saddle)
   - Sweeps theta outward from the cusp vertex
   - At each theta, compares `pearcey_cusp_term` (with the table) vs exact engine
   - Finds the angular distance where relative error exceeds the F016 bar
   - Reports the MINIMUM reach across all tested cusps

3. Set `_CUSP_ARM_COVERAGE` to the measured minimum reach.

4. Add `cogwheel/data/pearcey_table.npz` to the repo.

## Acceptance

- `pearcey_table.npz` exists and loads correctly via `use_pearcey_table`.
- `_CUSP_ARM_COVERAGE > 0` (arm enabled).
- At the coverage boundary, served Pearcey values agree with exact engine
  within the F016 envelope bar.
- Census shows cusp-neighbourhood draws served (not all falling through).

## Constraints

- The table generation may be slow (Pearcey function evaluation). Budget
  up to 5 minutes for table generation.
- Fast tests only for the build gate.
- Follow AGENTS.md and the spec/TODO workflow.
