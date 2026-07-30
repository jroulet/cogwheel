# Architect Short-Term Observations

## Build: analytic-cusp-serving-1c (2026-07-29)
Step 1c: (A) replace _pearcey_cusp._cusp_vertex FD-scan(129 pts ~258 calls)+golden-
section with analytic root of caustic_speed=0; (B) extend geometry.caustic_derivatives
to y'''. Prof rulings: FRAME — root-find g=y'.y'' in phase=theta-beta frame (caustic_
derivatives is beta-free/rotation-invariant), seed_phase=nearest.theta-beta, theta_cusp
=root+beta THEN critical_point(gamma,theta_cusp,beta,kappa,branch) [do NOT feed phase in,
it re-subtracts beta]; carry SAME branch end-to-end. BRACKETING — pos parity astroid
cusps EXACT at phase{0,pi/2,pi,3pi/2}: snap to nearest of 4 + confirm brentq in +-0.1;
saddle deltoid cusps = wedge-tip phase_c in{0,pi} (finite) + two wedge-EDGE cusps at
phase_c+-theta_max, theta_max=0.5*arcsin(lam/|gamma|) where caustic_derivatives DIVERGES
(LensDomainError). Pick candidate nearest seed; wedge-tip->short march+brentq strictly
inside wedge (_CUSP_BRACKET_EPS); wedge-EDGE->return None (named refusal, correct: old FD
straddled divergence & served finite-but-meaningless vertex). TWIN GATE: g(lo)<0<g(hi) AND
caustic_speed(root)<eps_speed*speed_scale, speed_scale=max caustic_speed(phase_c+-0.05..0.1
off-cusp), eps_speed=1e-4 (measure ratio, tighten if any real cusp in[1e-6,1e-4]); gate
fail->None (serve contract, NOT training's keep-sampled-min). Y''' TOL: mpmath.diff order3
@40dps, MEASURE worst, assert mixed atol_3=max(1e-10,3x worst_abs) rtol_3=max(1e-9,3x
worst_rel); cross-check 40->60 dps to separate mpmath noise (dps-sensitive, point-scatter)
from wrong closed form (dps-insensitive, systematic curve); stage-1 curve pin already in
CurveDefinitionStageOneTestCase (<=1e-13). SERVED-VALUES gate (acceptance#2, LOAD-BEARING):
Prof PRIMARY=vertex-INSENSITIVITY (perturb analytic vertex angle by {+-0.0245,+-1e-3,+-1e-4}
rad, assert |dF|/max|F|<envelope_bar on COMPLEX F); SECONDARY=reimpl old FD finder as oracle,
monkeypatch _pearcey_cusp._cusp_vertex, compare, EXCLUDE wedge-edge-divergence configs
(old serves wrong number there, new refuses = correct improvement). Configs: pos gamma
{0.05,0.3,0.6}, saddle {1.02,1.3}(+0.9@kappa0.3), kappa{0,0.3}, beta{0,0.37,1.1}, source in
cusp nbhd R in[1.2,5]*R_min both sides, w{20,40,80}; anti-vacuity >=60% served finite.
API (Simplifier): NOT order-kwarg (breaks 2-tuple unpackers). Extract private
_caustic_cascade returning all-order scalars; caustic_derivatives keeps 2-tuple assembly
LITERALLY identical (byte-id, oracle re-validates 1e-13); new public caustic_third_derivative
assembles y'''. Simplifier: single confirm-brentq path leaner than snap-only but Prof wedge
refusal is domain-necessary -> one fn, parity-gated analytic candidate list. Tests -> Test
Developer AUGMENT test_lensing_airy_fold.py (cusp_amplification lives there) + add direct
_cusp_vertex test (vertex.source==cusp loc, O(1) call-count) + y''' -> extend
test_lensing_caustic_derivatives.py. NOT Coder-authored. SPEC row53 geometry public-name
list gains caustic_third_derivative + verify _cusp_vertex prose still true -> post-gate
Librarian. WP1(_cusp_vertex) & WP2(y''') independent files, parallel OK.


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
