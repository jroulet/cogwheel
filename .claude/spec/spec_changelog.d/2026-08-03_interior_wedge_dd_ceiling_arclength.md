---
bump: patch
---

### InteriorWedgeChart: DD-product w-ceiling and arc-length angular axis

Commit 56a223a added two features to `InteriorWedgeChart.from_wedge_engine`
in `cogwheel/lensing/surrogate.py`. SPEC.md row 56 (sampling/surrogate layer)
updated to reflect both.

**Feature 1 — DD-product w-ceiling:**
`from_wedge_engine` now caps `w_max` at `_DD_PRODUCT_MARGIN / (r_grid[-1] *
reach_max)` before building the log-w grid, where `reach_max` is the maximum
caustic reach over theta nodes in the tile's `theta_wedge_range` (read from
`_WedgeCausticMap`). Every training node therefore satisfies `w * r *
r_caustic < _DD_PRODUCT_MARGIN = 58` (the engine's DD-product ceiling) by
construction. The cap is applied before the density calculation so node
spacing is unaffected.

**Feature 2 — Caustic arc-length angular axis:**
`from_wedge_engine` always builds a `theta_to_s` map — a `(2, 2001)` array
`[theta_fine, s_fine]` where `s_fine` is the cumulative-trapezoid integral of
`geometry.caustic_speed` at the tile's median gamma over 2001 equally-spaced
nodes in `theta_wedge_range` (same density as `_FARFIELD_ARC_MAP_SIZE` and the
tube's `_TUBE_ARC_MAP_SIZE`) — and derives `s_grid` as the images of the
training `theta_wedge_grid` nodes through that map via `np.interp`. The
spline's 4th axis is `s_grid` rather than raw `theta_wedge`, making knot
density arc-length-uniform across the wedge. At serve time a query
`theta_wedge` is converted to `s` through the stored map (one `np.interp`,
no quadrature in the hot path).

Previously SPEC.md stated only "Optional `theta_to_s` (shape `(2, N_map)`)
reparametrises the `theta_wedge` axis to an arc-length coordinate at serve
time." — this was accurate for the class API but did not document that
`from_wedge_engine` always constructs the map, the map size (2001 nodes), the
construction method (cumulative trapezoid of `caustic_speed`), or that `s_grid`
is derived from the map and used as the spline axis.
