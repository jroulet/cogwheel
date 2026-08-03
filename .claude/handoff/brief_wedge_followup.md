# Build Brief: InteriorWedgeChart — r-dependent w-ceiling + arc-length axis

## Mission

Fix two issues with the InteriorWedgeChart landed in ff06b8a:

1. **r-dependent w-ceiling**: `from_wedge_engine` currently uses a FLAT
   w-range for all (r, theta) grid points, causing 173/180 points to be
   refused by the engine (DD product violation at large r × high w). The fix:
   at each (r, theta) grid point, cap w_max at DD_MARGIN / (r × reach),
   where reach = r_caustic(gamma, theta). This is the core innovation of the
   wedge architecture — the grid naturally respects the DD constraint.

2. **Arc-length angular coordinate**: The angular axis currently uses raw
   `theta_wedge = atan2(|y2|, |y1|)` (source polar angle). It should use
   arc length along the caustic — the ant-crawling coordinate from cusp to
   cusp. The arc-length map is ALREADY computed and stored in every far-field
   chart's `arc_map` (`_FarFieldArcMap.theta_fine`, `s_table`). Reuse it
   as the wedge chart's `theta_to_s` reparametrization.

## Implementation

### r-dependent w-ceiling

In `from_wedge_engine`, replace the flat `log_w_grid = _log_w_grid(w_range, wnpd)`:
- Keep a global `w_min` (the low end, same for all points)
- At each (r, theta_wedge) point, compute `w_max_local = DD_MARGIN / (r * reach_at_theta)`
- Use the per-point `w_max_local` to determine which w-nodes to evaluate
- The stored `log_w_grid` on the chart is the UNION of all w-nodes used;
  points that don't extend to the full w_max get NaN/zero padding or
  a ragged structure
- OR simpler: use a GLOBAL w_max set by the MINIMUM r in the grid:
  `w_max_global = DD_MARGIN / (r_min * reach_max)`. This is conservative
  but avoids ragged arrays. At r_min=0.1, reach~1.4: w_max = 58/(0.1*1.4) = 414.
  That covers all draws.

### Arc-length angular axis

- In `from_wedge_engine`, after building `theta_wedge_grid`, compute the
  arc-length map via `_caustic_arclength_map` (or `_tube_arc_length_map`)
  at the representative gamma.
- Store as `theta_to_s` on the chart (same field as LobeInteriorChart).
- At serve time, `_evaluate_chart` remaps theta_wedge → s via `np.interp`.
- The existing far-field charts' arc_map can be referenced for consistency
  but the wedge chart builds its own (covers [0, π/2] not a single arc).

## Out of scope

- Full production training
- Census verification (do that after this lands)

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
