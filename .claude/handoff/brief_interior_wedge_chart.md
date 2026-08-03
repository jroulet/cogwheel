# Build Brief: Caustic-Relative Interior Chart Coordinates

## Mission

Replace the current interior chart architecture (which uses a rectangular
grid in absolute (y1, y2) coordinates, hitting the DD product cap at high w)
with caustic-relative coordinates that naturally respect the DD constraint
and allow the chart to cover the full w-range needed by all prior draws.

## The problem

The DD product cap (`w × |y| < 58`) limits chart w_max to ~9-22 depending
on the spatial extent of the grid. A rectangular grid in (y1, y2) has
|y| up to `reach_max ≈ 1.87` at ALL w-nodes, so w_max is capped at ~31.
But draws at high mass need w up to 108 — and those draws have SMALL |y|
(the prior ensures `w × |y| < 55`). The rectangular grid wastes budget on
(high-w, large-|y|) corners that no draw ever visits.

## The fix: caustic-relative interior coordinates

Use the 4-fold symmetry of the astroid. One quadrant (between two adjacent
cusps) tiles the full interior via reflections. In that wedge:

1. **Radial coordinate `r`**: distance from the origin toward the caustic,
   normalized by the caustic reach along that direction. `r=0` is the center,
   `r=1` is the caustic. This directly controls `|y| = r × reach(direction)`,
   so the DD product at each grid point is `w × r × reach` — known exactly.

2. **Angular coordinate along the fold**: parameterizes which cusp-to-cusp
   arc direction the radial ray points toward. By 4-fold symmetry, only one
   wedge (0 to π/2 in the eigenframe) is needed.

In these coordinates:
- The grid is rectangular in (r, angle, w)
- At each (r, angle), the safe w_max = `DD_MARGIN / (r × reach)` is known
- Near the center (r → 0): w_max → ∞ (ppGO is accurate, high w is safe)
- Near the caustic (r → 1): w_max is finite but the Airy fold handles it
- The chart's w-range can be r-DEPENDENT: wide at small r, narrow at large r

This eliminates the DD bottleneck entirely — the chart covers whatever w
each draw needs, because the grid only samples (r, w) combinations that
are DD-safe.

## Implementation approach

- A new chart type `InteriorWedgeChart` in surrogate.py (or modify the
  existing SACR-C infrastructure)
- Coordinates: (r, angle, w) where r ∈ [0, 1], angle ∈ [0, π/2]
- The w-axis is r-dependent: at each r, w_max(r) = DD_MARGIN / (r × reach)
- Spline the demodulated envelope in these coordinates
- Exploit 4-fold symmetry: train one wedge, serve all four by reflection
- At serve time: map (y1_eig, y2_eig) → (r, angle) → look up in the wedge

## Out of scope

- Training (step 9)
- The saddle interior (already has LobeInteriorChart)
- ppGO certification (this fix makes it unnecessary for interior)
- Crown band (high gamma — separate issue)

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
- Consult the Professor on the coordinate mapping and symmetry exploitation.
