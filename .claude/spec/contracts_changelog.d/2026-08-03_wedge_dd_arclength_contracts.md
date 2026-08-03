---
bump: patch
---

### InteriorWedgeChart: theta_to_s and DD-cap description updated

Updated `lens_amplification_surrogate` description to reflect the mandatory `theta_to_s` arc-length axis map (shape (2, 2001)) now always written by `from_wedge_engine`, the arc-length-uniform `s_grid` used as the spline's 4th axis, and the training-time w-axis cap enforced via `_DD_PRODUCT_MARGIN / (r_max * reach_max)`. Backward-compatible loading of pre-56a223a artifacts (absent theta_to_s) is noted.
