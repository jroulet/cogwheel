# Session: WP2(a) saddle rho additive form (lensing geometry)

## Context
cogwheel lensing extension (not core PE): `geometry.r_caustic(gamma, theta, kappa)`
and `_to_caustic_fixed` sampled<->standard transform for a gravitational-lensing
caustic coordinate. Macro-saddle branch (gamma>1) has DISCONNECTED deltoid lobes
that do NOT enclose the origin, so origin-referenced DIRECTIONAL r_caustic refuses
(raises LensDomainError) for off-wedge exterior rays.

## Ruling given
WP2(a) "switch to additive form" reconciled as: keep the SCALAR reach
`_caustic_reach(gamma)` (defined on every ray, no wedge dependence) but make the
combination ADDITIVE not multiplicative:
  forward:  rho = 1 + |y| - _caustic_reach(gamma)
  inverse:  |y| = _caustic_reach(gamma) + rho - 1
- Kills the multiplicative gamma/radius reach-stretch coupling (the brief's actual
  complaint).
- Gives drho/d|y| = 1 exactly.
- Round-trips on EVERY ray (no off-wedge refusal).
- Directional r_caustic(gamma,theta) is ill-posed here (proven by code: refuses
  off-wedge). Defer lobe-local radius to S2-2 per-lobe serve slice.

## Test guidance
1. rho>1 <=> outside-caustic consistency test is MOOT for the saddle: the serve-path
   certificate is eta/image_count from the geometry partition, NOT rho. rho is only a
   within-box coordinate. Drop the (rho>1) certificate test.
2. Required saddle tests instead:
   - round-trip to 1e-12
   - drho/d|y| = 1 exactly
   - REFUSAL-ABSENCE: no LensDomainError for any in-box exterior node across theta_c
     span at >=3 gammas>1. This is the physics content of choosing scalar reach; it's
     the property the directional form violates.
3. Grid: theta_c spanning [-pi,pi] AND gammas in (1,1.6]. Single theta_c or
   wedge-interior sample would hide a regression to directional-r_caustic (which passes
   on-wedge, raises off-wedge). Sample theta_c densely enough to land nodes off both
   deltoid lobes. Pair round-trip + refusal-absence on the SAME sweep.

## Note
WP1 per-column directional admission is parity==1 only; saddle exterior tiler stays on
existing scalar path. Saddle not trained/served yet.
