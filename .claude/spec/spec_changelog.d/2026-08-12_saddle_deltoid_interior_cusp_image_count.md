---
bump: patch
date: 2026-08-12
---

Update INTERIOR CUSP SERVING: interior discriminator changed from origin-based
`r_caustic` directional check to image-count gate `_is_interior = len(images) >= 4`.
This is the parity-correct discriminator for both astroid and deltoid — the deltoid
caustic does not enclose the origin, so `r_caustic` misclassified sources in the
corridor between the two lobes.  The generic interior case condition is now
`len(stationary_values) == 3`; the degenerate cluster condition is now
`len(images) >= 4, len(stationary_values) == 1`.  Exterior sources now described
as `len(images) < 4` (was `rho > 1`).
