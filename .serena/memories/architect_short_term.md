# Architect Short-Term Observations

## Build: arc-guard-fix-F041 (2026-07-29)
[re-plan 2026-07-29] Confirmed tree: _make_arc guard at line 689 still `abs(dot)<=0.1`;
_find_cusps sig = (thetas,speed,periodic,*,gamma,branch,width_safety=,min_halfwidth=)
so callers omitting gamma/branch fail at RUN. _branch_speed_profile STILL EXISTS (brief
"retired" list is loose; do NOT touch it). stable_gamma_bands returns (stable,dropped);
CausticStructure.arcs is the arc list field. Prof re-ruled (SUPERSEDES his own |dot|>0.1
verdict): exact-zero tripwire `if dot==0.0: continue` + sign from first evaluable frac IS
correct OPTION 1; NO gamma-stable ratio inside orientation after delete -> acceptance 2 =
arc EXISTENCE at gamma{0.02,0.1,0.3,0.9}; frac=0.5 sign == old guard on gamma>=0.1
(|dot|>=0.15 there); fallback-frac sign stable (two-image side is GLOBAL fold property).
Simplifier: 1 Coder WP lean, test shards lean/watch, docs->Librarian. PLAN = 1 Coder WP
(guard fix) + domain_test_descriptions split 2 suites (surrogate.py caller; training.py
callers+acceptance).
Finish crashed 1b: ONE production change in surrogate_training._make_arc.
FIX = Professor+Simplifier ruling OPTION 1 (delete guard): replace
`if abs(dot) <= 0.1: continue` with `if dot == 0.0: continue` (exact-zero
tripwire only, NOT a magnitude filter — |dot| ~ 1.5*gamma is fold TRANSVERSALITY
not cusp proximity, same category error as retired _PROBE_ETA). Keep the
LensDomainError fallback loop; sign taken at frac=0.5 (first evaluable), 12-orders
margin (min|dot|=4.4e-3), arc cusp-window-trimmed so |y'|/|y''|>=0.39 at midpoint.
Prof: image_count==4 is parity constant (astroid interior, no find_images);
soft_axis e->-e provably harmless; _tube_normal FD safe at gamma=0.02.
Tests -> Test Developer (NOT Coder): (a) new acceptance test bands+arcs; fix stale
_find_cusps callers test_lensing_surrogate.py ~L1068 + test_lensing_surrogate_training.py
~L1006/1660/1661 (add kw gamma/branch); delete any removed-constant pins.
Assert existence AND sign-stability (inward_sign identical across gamma set), NOT
|dot| magnitude. DOC-SYNC (SPEC row55, COVERAGE_DESIGN) = post-gate Librarian, flag only.

## Build: analytic-consumers-1b (2026-07-29)
Retire 6 numerical estimators in lensing/surrogate_training.py, re-express vs 1a
geometry fns (caustic_speed/caustic_curvature_radius/fold_opening_direction/
nearest_caustic_point). Targets: (1)_min_curvature_radius->min caustic_curvature_radius
endpoints INCLUDED (exact is 4.9-9.6% SMALLER, F038; do NOT assert byte-id/margin);
(2)_branch_speed_profile->caustic_speed directly, drop np.gradient+rolled CD;
(3)_find_cusps LOCATION only->brentq root of y'.y''=0 (caustic_speed touches 0, NOT
sign-changing, so root-find on y'.y''=d(speed^2)/2 which IS sign-changing at the min),
needs gamma+branch in signature; WINDOW delta_theta BYTE-IDENTICAL (keep 0.2 dip walk
inlined/renamed, _CUSP_SPEED_REL_FRAC name gone); (4)delete _probe_arc_side+_PROBE_ETA,
inward_sign=sign(fold_opening_direction . _tube_normal normal) [same normal serve uses],
image_count from parity (astroid served=4, saddle served=4? Prof confirm) or 1 find_images
at eta_max; (5)_caustic_inradius min|y| discrete->bracketed+refined min of closed form,
KEEP winding on cloud; (6)delete _CLOUD_MARGIN_FRAC, _InteriorAdmission.admits use
geometry.nearest_caustic_point (needs gammas field on dataclass), exterior stays byte-id.
OUT: cusp WINDOW width rule, _pearcey_cusp, ppgo. ACCEPT: eta_max>0.5*r_min flips on NO
band; stable_gamma_bands((0.01,0.30),+1) ZERO dropped slivers; no gradient/FD/step-const
in 6 targets. BLAST: test file WP3 byte-identity tests (_find_cusps sig + astroid frozen
copy) BREAK intentionally (cusp angles move to analytic root) -> Test Dev re-baseline.
CONTRADICTION to watch: delete _CUSP_SPEED_REL_FRAC name yet keep window byte-identical
(window uses the 0.2 threshold) -> inline 0.2 in window code, name gone.

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
