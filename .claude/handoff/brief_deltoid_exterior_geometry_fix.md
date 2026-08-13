# Build Brief: Fix deltoid exterior surrogate geometry — lobe-local coordinates, corridor rho, cusp-exclusion frame

## Mission

Fix the deltoid (saddle, gamma>1) EXTERIOR surrogate geometry, which is
origin-based but the deltoid's off-origin lobes make origin-polar
coordinates topologically wrong. Reuse the lobe-local machinery the
interior lobe charts already have. This is the "cherry on top" of the
recent saddle fixes (origin-rho ppGO, saddle interior serving). Documented
in `.claude/spec/todo.d/lensing_deltoid_exterior_geometry.md`.

## Measured facts (at HEAD 6a45e52)

1. **NEGATIVE rho in the corridor** (`surrogate.py` `_to/_from_caustic_fixed`):
   corridor source (0.5,0) at gamma=1.3 maps to rho=-0.214; `_from_caustic_fixed`
   raises `ValueError: rho must be non-negative`.  The scalar-additive exterior
   coordinate `y_mag = reach + rho - 1` cannot represent the between-lobes region.

2. **`_exclude_near_cusp` reconstructs the cusp at the WRONG position**
   (`surrogate_training.py`): uses `r_caustic(gamma, phi)*(cos,sin)` from the
   origin, but the actual cusp vertex (`geometry.critical_point`) is 0.108
   source-plane units away (gamma=1.3, beta=0.37).  And `r_caustic` fails for
   most deltoid angles (LensDomainError, 190/360 missing).

3. **The deltoid cusp is at a GENERAL source-plane (x, y)** (e.g. gamma=1.3:
   cusp at (-1.714, 0); gamma=1.5: (-1.739, -0.394) — coordinate-rotated so one
   cusp is on an axis, but at a general radial distance, NOT the origin).  The
   teardrop exclusion must be centered there.

4. **The exterior set is TOPOLOGICALLY disconnected along origin-rays**
   (measured gamma=1.3): ray at theta_c=0/0.3 passes EXT(2img) -> INT(4img
   lobe) -> EXT(2img).  A single origin-polar (rho, theta_c) cannot index the
   exterior (two disjoint pieces per ray); after excising cusp teardrops the
   remainder is not simply connected on the D2-folded quarter plane.

5. **Fundamental unit cell is not a clean quadrant**: 5 D2-folded deltoid cusp
   directions in [0, pi/2] (0.0 x2, 0.737 x3) vs the astroid's 2.

## The fix direction (Professor must adjudicate the exact construction)

Reuse the EXISTING lobe-local machinery the interior lobe charts already
use (all present in the code):

- `surrogate._from_lobe_fixed` / `_to_lobe_fixed` (surrogate.py:839/883) —
  lobe-centered `(rho_lobe, theta_lobe)` with centroid frame.
- `surrogate._lobe_boundary_radius` (surrogate.py:806) — `rho_lobe =
  |y-centroid| / r_deltoid(theta)`, the single authoritative deltoid
  boundary (handles the general cusp position naturally).
- `surrogate._lobe_cusp_axis_map` (surrogate.py:624) — cusp-adapted
  u = d^(2/3) axis.
- `surrogate_training._SaddleLobeAdmission` (surrogate_training.py:2389) —
  lobe admission predicate.

The deltoid EXTERIOR must be charted per-connected-component with a
lobe-centered exterior coordinate (rho_lobe outside the lobe), never
origin-polar.  The cusp-exclusion blob must be centered at the ACTUAL cusp
vertex (from `critical_point`), not reconstructed from origin r_caustic.

The Professor must decide the exact exterior rho convention (e.g.
`rho_lobe = |y-centroid| / r_deltoid(theta)` with rho_lobe>1 exterior,
matching the interior's rho_lobe<1), and how the corridor (between lobes)
is handled (it may be a separate connected component or the two lobe
exteriors plus the corridor).

## Acceptance
1. Deltoid corridor sources map to a VALID non-negative rho (no
   `ValueError`); the exterior chart coordinate is lobe-local and
   topologically well-posed.
2. The cusp exclusion is centered on the actual cusp vertex at its general
   (x, y), oriented correctly — not origin-r_caustic reconstructed.
3. The census saddle-interior gap drops from ~17% (re-run
   `scripts/census_dry_run.py --n-samples 10000` after the fix; note the
   census itself hardcodes saddle-interior as 'exact_engine' at
   census_dry_run.py:157-159, so ALSO update the census to model the
   lobe/cusp serve paths — otherwise it can't see the fix).
4. Astroid (positive parity) is UNCHANGED — all changes gated on gamma>1.
5. `test_lensing_surrogate.py`, `test_lensing_surrogate_lobe.py`,
   `test_lensing_surrogate_census.py`, `test_lensing_exterior_admission.py`,
   `test_lensing_surrogate_training.py` green.

## Constraints
- Fast tests only. Refusal-conservative (never serve a wrong lobe).
- Reuse `_from_lobe_fixed`/`_to_lobe_fixed`/`_lobe_boundary_radius`/`_SaddleLobeAdmission`
  — do NOT duplicate lobe geometry.
- The census probe (`scripts/census_dry_run.py`) is the measurement
  instrument; update it to model the saddle lobe/cusp serving paths so the
  17% gap is visible and closable.
- Professor adjudication required on the exterior rho convention + corridor
  treatment before the Coder codes.
