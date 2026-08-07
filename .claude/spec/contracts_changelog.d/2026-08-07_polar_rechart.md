---
bump: patch
---
Update the `lens_amplification_surrogate` description in DATA_CONTRACTS.yaml:
- Drop the "exterior positive-parity only" restriction on `ExteriorPolarChart`
  records -- macro-saddle (parity != 1) exterior charts are now chartable in
  the same caustic-fixed polar axes with an additive scalar-reach rho
  (`rho = 1 + |y| - _caustic_reach`, `drho/d|y| = 1`); the stale
  "remain exact-engine fall-through" sentence is removed.
- Clarify that rho is additive for the exterior (both parities) and
  multiplicative only on the astroid interior arm.
