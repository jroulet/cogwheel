---
date: 2026-07-30
---

### The caustic reach is now a closed form, not a 1440-point scan

`cogwheel.lensing.ppgo_map.caustic_geometry` returned the maximum source-plane
caustic radius by sweeping the critical curve over 720 polar angles on both
square-root branches, calling `geometry.critical_point` at each — 1440
evaluations per call, on every likelihood evaluation that serves through the
surrogate or the certified ppGO map.

The maximum has a closed form. Eliminating the polar angle in favour of
`u = 1 / ((1 - kappa) |x|^2)` makes the squared caustic radius a rational
function of `u` whose stationary condition factors exactly, leaving a finite
candidate set: the two axis cusps, the macro saddle's wedge turnaround, and
two interior roots that are real only for effective shear `>= sqrt(3)/2`. The
reach is the largest candidate that a real angle actually attains.

`critical_point` is now called **zero** times per reach, and the function costs
5.4 us where the scan cost 42.95 ms (~7900x, measured over 13 shear/convergence
configurations spanning both parities). Served values are unchanged; where the
new answer differs from the old, by up to 1.1e-4 relative in the narrow band
just above the parity wall, it is the retired scan that was wrong — its answer
converges to the closed form as the grid refines, because the extremum there
falls between grid nodes.

`caustic_geometry` no longer accepts an `n_theta` argument, and now raises
`LensDomainError` on `kappa >= 1` (over-critical) and on `|gamma| == 1 - kappa`
exactly (the `det A = 0` parity wall) rather than returning a degenerate reach.
