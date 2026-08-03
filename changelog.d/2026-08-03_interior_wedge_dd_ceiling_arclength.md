---
date: 2026-08-03
---

### InteriorWedgeChart: DD-product w-ceiling and caustic arc-length angular axis

`InteriorWedgeChart.from_wedge_engine` in `cogwheel/lensing/surrogate.py` gains
two new features:

**DD-product w-ceiling:** The training `w_max` is capped at
`_DD_PRODUCT_MARGIN / (r_grid[-1] * reach_max)` before the log-w grid is built,
where `reach_max` is the maximum caustic reach over theta nodes in the tile's
`theta_wedge_range`. Every training node therefore satisfies
`w * r * r_caustic < _DD_PRODUCT_MARGIN = 58` (the engine's diffraction-delay
ceiling) by construction; the cap is applied before the density calculation so
node spacing is unaffected.

**Caustic arc-length angular axis:** `from_wedge_engine` now always builds a
`theta_to_s` map — a 2001-node cumulative-trapezoid integral of
`geometry.caustic_speed` at the tile's median gamma — and derives `s_grid` as
the images of the training `theta_wedge_grid` nodes through that map. The
spline's 4th axis is `s_grid` rather than raw `theta_wedge`, making knot density
arc-length-uniform across the wedge. A query `theta_wedge` is converted to `s`
at serve time via one `np.interp` call (no quadrature in the hot path).
