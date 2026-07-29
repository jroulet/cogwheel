# Architect Short-Term Observations

## Build: analytic-geometry-cascade-1a (2026-07-29)
Add 4 analytic caustic fns to chang_refsdal/geometry.py beside r_caustic:
caustic_derivatives(gamma,theta,*,kappa=0,branch=1)->(y',y''); caustic_speed=|y'|;
caustic_curvature_radius=|y'|^3/|cross|; fold_opening_direction=unit D2y[e,e].
KEY: brief's cascade text `p=-u, p'=-u'` is a TYPO — Professor confirmed correct
is p=(lam-+gamma)-lam*u, p'=-lam*u', p''=-lam*u'' (the lam factor, wrong at kappa!=0).
Positive parity IGNORES branch (mirror critical_point: branch=-1 must NOT nan/warn).
fold dir: D2y[e,e]=(4(x.e)e+2x-8(x.e)^2 x/r^2)/r^4, return UNIT; e->-e harmless (Prof Q2).
DISCOVERY: cogwheel/tests/test_lensing_caustic_derivatives.py ALREADY EXISTS,
complete+correct (lam*u) for the 3 scalar fns w/ mpmath oracle, mixed tol
atol=5e-13+rtol=1e-11, astroid pin, oracle-independence AST guard, self-falsification.
GAPS (Test Dev must ADD): (1) STAGE-1 curve validation _oracle_y_component vs
critical_point.source <=1e-13 (existing file is stage-2 only — brief mandates 2-stage);
(2) positive-parity branch=-1 no-nan/no-warning test; (3) fold_opening_direction tests
(image-count both sides via find_images, resolvable pts per F039 31/32).
Coder WP1=cascade+2 scalars, WP2=fold dir (separate closed form, uses image/soft_axis).
SPEC row for geometry.py gains new public names -> post-gate Librarian; strike step-1
block from todo.d/lensing_analytic_derivatives.md + completion record -> Librarian.


(empty — last consolidated by Dreamer on 2026-07-28)

## Build: positive-parity-resolved-first (2026-07-28)
Unify geometric-vs-wave predicate on `select_branch` in operator.py's two grids.
- POS-PARITY (_positive_parity_grid): add geometric branch. For w>ceiling nodes:
  L=w*|y'| (cache |y'| from top-of-grid _mass_sheet_map y_scaled; do NOT re-call
  cancellation_exponent per node), delta_min via _real_delay_min_separation(
  physical source, macro_matrix) once, guarded by any(w>ceiling). geometric ->
  geometric_amplification (physical frame!), else existing arms-then-refuse.
- SADDLE (_saddle_grid): Professor OVERRULED my pi*w/4 idea AND Simplifier's
  rubber-stamp of it. pi*w/4 (DD-mantissa depth) vs L_MAX=48 (1F1 onset proxy) =
  unit-mismatch + opens dead band (60,61.115]. RULING: pass cancellation_exp=
  math.inf so only the resolution leg routes through select_branch -> byte-
  identical w>60 AND resolved boundary; ceiling stays enclosing branch.
- Residual: ~1% O(1) tail survives 2-condition gate (p99 7.1e-1, max 74); NEVER
  "certified/exact"; add FINDINGS entry.
- All tests -> Test Developer (new + re-point ~8 blast-radius files). Docs ->
  post-gate doc-sync/Librarian.
