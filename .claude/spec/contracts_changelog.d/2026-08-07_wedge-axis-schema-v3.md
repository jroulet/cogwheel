---
date: 2026-08-07
bump: major
---

### InteriorWedgeChart axis schema v2 -> v3: `theta_to_u` / `u_grid`, required map

The wedge chart's angular spline axis has been the cusp-adapted coordinate
`u = d**(2/3)` (`d` = angular distance to the near astroid cusp) since the
cusp-axis build, but it was serialized under arc-length field names
(`theta_to_s`, `s_grid`) and validated by the shared `_validate_theta_to_s`.
The names recorded the symbol's first use rather than its role: `u` is
`rad**(2/3)`, not a length.

Contract changes, WEDGE ONLY:

- `axis_schema` `'wedge_caustic_relative_v2'` -> `'wedge_caustic_relative_v3'`.
  v3 is the only known tag. v1 (genuinely arc-length) and v2 (the same `u`
  axis under the wrong field names) both hard-refuse at load.
- Serialized field `theta_to_s` -> `theta_to_u`; derived `s_grid` -> `u_grid`.
- `theta_to_u` is REQUIRED. It is read unconditionally at load; an absent map
  hard-refuses. The v2 backward-compatible tolerance for pre-56a223a artifacts
  is retired along with the v2 tag.

The tube, lobe-interior and far-field maps genuinely hold arc length and are
UNCHANGED: they keep `theta_to_s` and `_validate_theta_to_s`. Both validators
now delegate to a shared `_validate_axis_map` core enforcing monotonicity and
start-at-zero only, with deliberately no length-scale bound — `u` and `s` have
different magnitudes over the same tile, so neither may inherit the other's.

BREAKING for stored artifacts: every wedge chart written before this bump
fails to load by design, rather than being served on a mislabelled axis. Any
wedge chart trained during the 2026-08-06 coordinate probes must be retrained.
Served `F` is unchanged — serve is coordinate-agnostic through the stored map.

Flagged three times by the Inspector during the `subdivision_recursion` build
(INS-1-001) and correctly routed to doc-sync each time; applied here by the
driver after the tree gate blocked the build's own Librarian phase.
