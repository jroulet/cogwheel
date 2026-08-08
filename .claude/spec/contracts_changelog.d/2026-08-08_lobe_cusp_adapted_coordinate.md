---
date: 2026-08-08
bump: major
---

### LobeInteriorChart axis schema -> `lobe_caustic_relative_v1`: `theta_to_u` / `u_grid`, required map

The lobe-interior chart's angular spline axis is now the cusp-adapted
coordinate `u = d**(2/3)` (`d` = angular distance to the nearest deltoid
cusp vertex), the macro-saddle counterpart of the wedge v3 axis. The
sqrt-edge `s` coordinate is deleted, and the field names record the
coordinate's role rather than its first use.

Contract changes, LOBE CHARTS ONLY:

- `axis_schema` tags `'lobe_local_offset_rholobe_thetalocal_framewinv'`
  (V1, raw `theta_local`) and
  `'lobe_local_offset_rholobe_thetalocal_sqrtedge_framewinv'` (sqrt-edge
  `s` coordinate) are both DROPPED. The ONLY known tag is
  `'lobe_caustic_relative_v1'` (`_LOBE_AXIS_SCHEMA_NEW`); an absent or
  unknown tag hard-refuses at load.
- Serialized field `theta_to_s` -> `theta_to_u`; the derived spline axis is
  `u_grid` (images of the training `theta_local_grid` nodes through the map).
- `theta_to_u` is REQUIRED, shape `(2, 2001)` = `[theta_fine, u_fine]`,
  built UNIFORM in `u` by `_lobe_cusp_axis_map`. It is read unconditionally
  at load; an absent map hard-refuses (KeyError). The V1 backward-compatible
  tolerance for absent maps is retired along with the V1 tag.

The tube and far-field maps genuinely hold arc length and are UNCHANGED:
they keep `theta_to_s` / `_validate_theta_to_s`. Both `_validate_theta_to_u`
(wedge AND lobe) and `_validate_theta_to_s` delegate to the shared
`_validate_axis_map` core with no length-scale bound.

BREAKING for stored artifacts: every lobe chart written before this bump
fails to load by design, rather than being served on a mislabelled axis.
Served `F` is unchanged — serve is coordinate-agnostic through the stored
map.
