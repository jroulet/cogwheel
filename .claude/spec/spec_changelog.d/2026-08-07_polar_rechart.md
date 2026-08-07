---
bump: patch
---
Update the GLOBAL MULTI-CHART ARTIFACT description and the Key-abstractions
far-field surrogate coordinate contract in SPEC.md:

- The surrogate collection description now names `ExteriorPolarChart` and
  its caustic-fixed polar `(rho, theta_c)` axes explicitly (previously
  "exterior-polar charts" without the class name).
- The Key-abstractions contract now covers BOTH parities: the "Macro-saddle
  exterior charts remain exact-engine fall-through" sentence is replaced —
  macro-saddle exteriors are chartable in the same polar coordinates, with
  an additive scalar-reach `rho` (`rho = 1 + |y| - _caustic_reach`,
  `drho/d|y| = 1`), matching `_build_farfield_chart` and `_to_caustic_fixed`.
