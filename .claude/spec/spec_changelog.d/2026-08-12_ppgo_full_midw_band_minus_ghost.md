---
date: 2026-08-12
bump: patch
---
### Pearcey ppGO rung full mid-w band + MINUS_GHOST saddle exterior window

SPEC.md Microlensing engine row (ppGO rung description): the rung is no
longer described as "high-w" — `_R_PPGO_ERROR_CONST = 0.10` (measured,
`r_ppgo_min ~ radius_min ~ 7.37`) makes it fire across the FULL mid-w band
(previously only deep-asymptotic `R` served, leaving the mid-w window to an
exact-engine flashback on the astroid and a refusal on the deltoid);
measured rel-err 5e-6..6e-5 across the opened band, both parities.

Born / `_surrogate_coefficients` paragraph: added the saddle EXTERIOR CUSP
WINDOW serving — `ExteriorPolarChart` under the `FARFIELD_KERNEL_SUM_
MINUS_GHOST` envelope label (ghost gates become the admission authority;
`_CUSP_EXCLUSION_DISTANCE` reduced for saddle near-cusp tiles), closing the
exterior cusp neighbourhood quadrature-free on both parities.
