---
bump: patch
---

### InteriorWedgeChart: document s_grid axis and DD-product w-ceiling in record format

`DATA_CONTRACTS.yaml` `lens_amplification_surrogate` InteriorWedgeChart
description updated for commit 56a223a:

**Arc-length axis map (theta_to_s / s_grid):**
Prior entry said "optional theta_to_s shape (2, N_map)". Now clarified:
- `theta_to_s` shape is always `(2, 2001)` when built by `from_wedge_engine`
  (row 0: `theta_wedge_fine`; row 1: `s_fine` = cumulative trapezoid of
  `geometry.caustic_speed` at median gamma).
- `s_grid` (shape `(n_theta_wedge,)`) is the images of the training
  `theta_wedge_grid` through the map; the B-spline 4th axis is `s_grid`,
  NOT raw `theta_wedge`. This is not a separate serialized field — the
  spline knots encode the s-values; `s_grid` is a `from_wedge_values`
  argument, not a chart attribute.
- `theta_to_s` absent only for pre-56a223a artifacts; `from_wedge_engine`
  always builds and stores it.

**DD-product w-ceiling:**
`from_wedge_engine` clips `w_max` to `_DD_PRODUCT_MARGIN / (r_max *
reach_max)` before the log-w grid is built. Not a separate serialized
field, but recoverable from `log_w_grid[-1]`.
