---
bump: patch
---

Doc sync: the F070 low-end `w`-floor clamp paragraph described the serve-side
guard as a proposal ("the guard is free at the serve site... "). It shipped
same day in 8dfb8ca: `LensedRelativeBinningLikelihood` now re-checks
`farfield_w_floor(geom.delays, geom.real_mask)` against the served chart's
sub-band bottom for every `_FARFIELD_KERNEL_FAMILY` label and refuses below
it. SPEC.md now states this as shipped. The training-side half of the gap —
no `FARFIELD_DIFFRACTIVE` tile is ever trained for the sub-floor band, so
production refuses rather than covers — remains open
(`todo.d/lensing_low_mass_exterior_training_registers_zero_charts.md`); no
content change needed there beyond the pointer already added.
