---
section: Backlog
depends_on: []
---

- **Deltoid EXTERIOR surrogate geometry uses origin-based coordinates, but the cusp is at a general source-plane (x, y)**
  `[→ spec]` — measured 2026-08-12 by the driver (user-flagged).  Census
  probe finds 1742/10000 (17%) saddle-interior gaps — likely these.

  ## The core issue (user's point)

  The deltoid cusp vertex is at a GENERAL source-plane position (x, y) —
  e.g. gamma=1.3: cusp at (-1.714, 0.000); gamma=1.5: cusp at (-1.739,
  -0.394).  (The code coordinate-rotates so one cusp sits on an axis, but
  the cusp is at a general radial distance, NOT at the origin.)  The
  teardrop cusp-exclusion blob must be CENTERED at that general (x, y) in
  the source plane.  Any origin-based coordinate (origin-polar rho/theta_c,
  origin-ray r_caustic) is fundamentally mismatched to it.

  The INTERIOR lobe charts already do this correctly: lobe-local
  `rho_lobe = |y - centroid| / r_deltoid(theta)` (surrogate.py
  `_lobe_boundary_radius`), which naturally handles the general cusp
  position.  The EXTERIOR side does NOT.

  ## Confirmed bugs (measured at gamma=1.3)

  1. **`_to/_from_caustic_fixed` (surrogate.py) gives NEGATIVE rho in the
     corridor**: source (0.5,0) maps to rho=-0.214; `_from_caustic_fixed`
     raises `ValueError: rho must be non-negative`.  The scalar-additive
     exterior coordinate `y_mag = reach + rho - 1` cannot represent the
     between-lobes region (same origin assumption as the ppGO bug fixed in
     288f37c, but NOT fixed in the surrogate coordinate map).

  2. **`_exclude_near_cusp` (surrogate_training.py) reconstructs the cusp
     at the WRONG position**: it uses `r_caustic(gamma, phi)*(cos,sin)`
     from the origin, but the actual cusp vertex (`critical_point`) is
     0.108 source-plane units away (gamma=1.3, beta=0.37).  The teardrop
     exclusion is centered wrong.  And `r_caustic` fails for most deltoid
     angles (LensDomainError, 190/360 missing), so it can fail outright.

  3. **Fundamental unit cell is not a clean quadrant**: 5 D2-folded deltoid
     cusp directions in [0, pi/2] (0.0 x2, 0.737 x3) vs the astroid's 2.

  ## Fix direction

  The deltoid EXTERIOR needs the same lobe-local frame the interior
  already uses: `rho_lobe = |y - centroid| / r_deltoid(theta)` for the rho
  map, and the cusp exclusion blob centered at the actual cusp vertex
  source position (from `critical_point`), not reconstructed from origin
  `r_caustic`.  This is the same machinery as `_lobe_boundary_radius` /
  `_from_lobe_fixed` — extend it to the exterior.

  ACCEPTANCE: deltoid corridor sources map to a valid non-negative rho;
  the cusp exclusion is centered on the actual cusp vertex at its general
  (x, y); the census saddle-interior gap drops from ~17%.

  ## The exterior set is TOPOLOGICALLY disconnected along origin-rays (user, verified)

  Measured (gamma=1.3): an origin-ray at theta_c=0 or 0.3 passes
  EXT(2img) -> INT(4img lobe) -> EXT(2img) as rho grows.  The deltoid
  lobes are off-origin, so the exterior is TWO disjoint pieces along a
  ray, separated by a lobe-interior.  A single origin-polar (rho, theta_c)
  coordinate cannot index the exterior: the same ray has exterior at two
  disjoint rho-intervals, and after excising the cusp teardrops (at
  general (x,y)) the remainder is NOT simply connected on the D2-folded
  quarter plane and NOT ray-addressable from the origin.

  This makes origin-polar coordinate for the deltoid exterior
  topologically ill-posed, not merely imprecise.  The interior lobe charts
  avoid it with lobe-centered `rho_lobe = |y-centroid|/r_deltoid(theta)`
  (each lobe is a connected set).  The EXTERIOR must be charted per
  connected component with a lobe-centered exterior coordinate, never
  origin-polar.
