---
date: 2026-08-08
bump: minor
---

### Lobe-interior charts use the cusp-adapted `u = d**(2/3)` angular coordinate

The `LobeInteriorChart` paragraph of the microlensing-surrogate section is
updated to the shipping contract. Lobe charts are now charted in lobe-local
`(rho_lobe, u)` where `u = d**(2/3)` with `d` the angular distance to the
nearest deltoid cusp vertex — the exact, gamma-universal caustic-reach cusp
scaling, eliminating the `|dtheta|**(1/3)` divergence the raw `theta_local`
axis carried at a cusp (mirroring the wedge v3 cusp-adapted axis).

The old two-tag description is replaced by the single axis-schema tag
`_LOBE_AXIS_SCHEMA_NEW = 'lobe_caustic_relative_v1'` with the cusp-adapted
`theta_to_u` map REQUIRED: `from_lobe_engine` builds it via
`_lobe_cusp_axis_map` (a uniform-in-`u` `(2, 2001)` array
`[theta_fine, u_fine]`), and the loader reads it unconditionally, so an
absent map hard-refuses. Both OLD lobe tags — `_LOBE_AXIS_SCHEMA_V1` (raw
`theta_local` spline) and the sqrt-edge `_LOBE_AXIS_SCHEMA` (`theta_to_s`
array) — are DROPPED from the known set and hard-refuse at load.

Also stated: gated lobe tiles are subdivided at the U-MIDPOINT (mapped back
to `theta_local` through the nearest cusp's adapted map), not the raw theta
midpoint, and the `_LOBE_CUSP_EXCLUSION_DISTANCE` carve-out is retired. The
field-name split in the wedge CUSP-ADAPTED ANGULAR AXIS paragraph is
corrected: the wedge AND lobe-interior maps carry `theta_to_u` /
`_validate_theta_to_u`; only the tube and far-field maps keep `theta_to_s`.
