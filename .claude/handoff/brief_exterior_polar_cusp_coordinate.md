# Build Brief: Exterior polar chart — `u = d^(2/3)` angular coordinate

## Mission

Replace `theta_c` in `ExteriorPolarChart` with `u = d^(2/3)` where `d` is angular distance to the nearer cusp (0 or π/2) in the D₂-folded quadrant. Same pattern as `InteriorWedgeChart`'s `u` coordinate (wedge v3, F064). Fixes the `d^{-1/3}` divergence in `dE/dtheta_c` near cusp angles that causes aggressive subdivision.

Positive parity (astroid, gamma < 1) only. Saddle exterior uses scalar rho without directional r_caustic — no change needed.

## Work

1. **Add `theta_to_u` map** to `ExteriorPolarChart`: optional `np.ndarray` attribute (shape `(2, N)`, row 0 = `theta_fine`, row 1 = `u_fine`). Reuse `_wedge_cusp_axis_map` helper from `surrogate.py:533` — same `origin='low'`/`'high'` convention.

2. **Update `from_engine`/`from_values`**: when `theta_to_u` is provided, fit the B-spline on uniform `u_grid` instead of uniform `theta_c_grid`.

3. **Update serve path**: `theta_c → u` via `np.interp` before spline contraction — identical to wedge path at `surrogate.py:2583-2588`.

4. **Update tiler** (`_farfield_exterior_tiles`): tiles are already cusp-aligned via `_cusp_aligned_theta_tiles` — each tile lives entirely on one side of the waist. Thread `theta_to_u` through `_build_farfield_chart`. Each tile gets the map for its `origin`.

5. **New axis schema**: `'exterior_polar_rho_u_v1'` (replaces `'exterior_polar_rho_theta_c'`).

6. **Update tests**: migrate exterior polar tests to `(rho, u)` coordinate.

## Measured facts (SHA 9597a4e)
- `InteriorWedgeChart.theta_to_u` at `surrogate.py:537` — proven pattern (F064: 171× improvement)
- `_wedge_cusp_axis_map` at `surrogate.py:533` — reusable helper
- `_farfield_exterior_tiles` at `surrogate_training.py:1923` — already cusp-aligned
- `_EXTERIOR_POLAR_AXIS_SCHEMA = 'exterior_polar_rho_theta_c'` at `surrogate.py:260`
- F064 measured: 171× better than adding nodes in the bad coordinate

## Constraints
- Fast tests. Follow AGENTS.md.
- Macro-saddle exterior unchanged
- `rho` stays as-is (`drho/d|y| = 1`, well-behaved)
- Mirror the wedge v3 pattern exactly — reuse helpers, same serialization

## Plan-gate requirement (rejected once: 2026-08-08)
The plan verification gate enforces DISJOINT test-suite write ownership:
each `domain_test_descriptions` spec names exactly ONE `test_*.py` primary
file (its owning suite), and a spec must NEVER reference another spec's
primary file. A prior plan was rejected because a shard owning
`test_lensing_surrogate.py` also named `test_lensing_exterior_windows.py`
(another shard's suite). Split the test work so each test file is listed in
exactly one spec — do not cross-reference suites between specs.
