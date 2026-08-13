---
date: 2026-08-12
bump: patch
---

Add the `LobeExteriorChart` record to the `lens_amplification_surrogate`
description (build 4c7dc92, WP2, deferred via INS-5-001). It is the exterior
sibling of `LobeInteriorChart`: same lobe-local `(rho_lobe, theta_local)`
frame, exterior shell `rho_lobe` in `(1, rho_outer]`, same
`lobe_caustic_relative_v1` axis-schema tag, NPZ kind tag `lobe_exterior`,
`image_count = _MACRO_SADDLE_EXTERIOR_IMAGE_COUNT = 2`, `FARFIELD_KERNEL_SUM`
envelope label, no `other_centroid` / `corridor_half` / fold-carrier fields,
and an optional cusp-adapted `theta_to_u` map read via a soft `data.get`.

Corrected the macro-saddle-exterior sentence that described it as
`ExteriorPolarChart`-chartable with an additive scalar-reach `rho`:
`ExteriorPolarChart` charts positive parity (astroid) only now: the two
deltoid lobes sit off the origin and neither encloses it, making an
origin-polar coordinate topologically ill-posed for the saddle exterior.
