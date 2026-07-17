---
date: 2026-07-17
---

### Lensing: certified the exact ``w -> 0`` macro-magnification limit (Build 2d)

The flat, mass-independent ``|F| - 1`` the Chang-Refsdal operator reports at
tiny ``w`` was confirmed to be the EXACT geometric-optics limit
``F(w -> 0) = 1/sqrt((1 - kappa)**2 - gamma**2) = sqrt(mu_macro)``, not a
``gamma/(2w)`` prefactor singularity. ``F`` is normalized to no lens at all,
so as the point-mass diffraction switches off the quadratic macro potential
integrates exactly (a Gaussian), leaving a real, frequency- and
mass-independent constant that is not ``1`` under shear or convergence. No
engine code changed: a proposed small-``w`` short-circuit returning
``1 + O(w)`` was rejected as a 2% discontinuity that would destroy the exact
pure-shear limit. A new closed-form gate
(``MacroMagnificationLimitTestCase``) pins ``|F_op|`` to the literal
``1/sqrt((1 - kappa)**2 - gamma**2)`` across a positive-parity grid and three
decades of tiny ``w``, and the lensing likelihood's zero-noise anchors were
re-based on this understanding. See FINDINGS F009.
