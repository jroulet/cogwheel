---
date: 2026-08-08
---

### Lobe interior chart adopts the cusp-adapted `u = d**(2/3)` angular axis

`LobeInteriorChart` (macro-saddle interior) replaces its sqrt-edge angular
coordinate with the cusp-adapted `u = d**(2/3)`, mirroring `InteriorWedgeChart`
v3. The 2/3 exponent absorbs the `r_deltoid ~ const - c*d**(2/3)` caustic-reach
power law at deltoid cusp vertices, removing the `|dtheta|**(1/3)` singularity
the raw `theta_local` axis left there. The single axis-schema tag is now
`lobe_caustic_relative_v1` with a required `theta_to_u` map (built via
`_lobe_cusp_axis_map`, uniform-in-`u`); both old lobe tags hard-refuse at load.
Gated tiles subdivide at the u-midpoint, and the `_LOBE_CUSP_EXCLUSION_DISTANCE`
carve-out is retired — the cusp-adapted coordinate handles near-cusp tiles
directly. 149 tests pass (0 fail).
