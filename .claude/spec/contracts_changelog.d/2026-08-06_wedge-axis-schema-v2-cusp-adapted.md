---
bump: patch
---

### InteriorWedgeChart angular axis schema bumped to v2 (cusp-adapted, not arc length)

`_WEDGE_AXIS_SCHEMA` moved from `'wedge_caustic_relative_v1'` to
`'wedge_caustic_relative_v2'` (`cogwheel/lensing/surrogate.py`). v1 is
DROPPED from `_KNOWN_WEDGE_AXIS_SCHEMAS`, not migrated: a stale v1
(arc-length) artifact hard-refuses at load rather than being served through
the new coordinate.

The angular spline axis stored in each record's `theta_to_s` / `s_grid`
fields is no longer caustic arc length `s = integral caustic_speed dtheta`.
It is the cusp-adapted coordinate `u = d**(2/3)`, where `d` is the angular
distance to the NEAR astroid cusp (`theta_wedge = 0` or `pi/2`). The wedge
is split at the per-band caustic waist `theta_waist = argmin_theta
r_caustic(gamma, theta)` (new helper `_wedge_theta_waist`); each tile's
angular span lies entirely on one side of the waist and its map
(`_wedge_cusp_axis_map`) uses the matching near-cusp origin. This absorbs
the `r_caustic ~ const - c * d**(2/3)` cusp scaling that made the retired
arc-length axis singular near the cusps.

Field names on disk (`theta_to_s`, `s_grid`) are unchanged; only what they
encode changed. Updated `DATA_CONTRACTS.yaml`'s `lens_amplification_
surrogate` description accordingly. Did NOT touch the separate
interior-coverage sentence (astroid interior "nominal domain" / currently
UNSERVED per `todo.d/lensing_wedge_angular_axis_is_cusp_singular.md`, which
consolidates the original `lensing_wedge_charts_fail_the_eps_bar` measurement)
-- that measurement claim is unrelated to the schema-tag correction and is
not yet demonstrated at production scale under the new coordinate.
