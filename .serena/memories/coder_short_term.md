# Coder Short-Term Observations

- INS-1-001 RESOLVED (F041 fix-build, test-file finding; pipeline routed
  stale-fixture retirement to Coder). In test_lensing_caustic_cusps.py
  deleted the 3 pre-fix-pathology pins: (1) whole class
  StableGammaBandsSliverTestCase (pin (b): dropped astroid-onset sliver no
  longer occurs post-F041); (2) FootOfNormalCurvatureValueTestCase::
  test_brief_small_bands_have_no_band_wide_arc (small bands now build arcs).
  Removed now-orphaned constants EXPECTED_DROPPED_SLIVER, STABLE_BAND_RANGE,
  FOOT_NO_ARC_BAND, FOOT_METAMORPHOSIS_BAND (grep-confirmed unused after
  deletions). Kept FOOT_FALSE_BANDS + FOOT_TRUE_BAND (still used by the two
  surviving FootOfNormal tests). Updated module docstring: dropped the
  no-arc/metamorphosis sentences from the FootOfNormal bullet + removed the
  StableGammaBandsSliver bullet, added a one-line note that F041 retired
  those pre-fix pins. ast.parse OK; pytest --collect-only = 25 tests, clean
  import (did NOT run test bodies; role boundary — Inspector runs the suite).
  OWED -> Test Dev / Inspector: SelfFalsificationTestCase::
  test_below_floor_alignment_fails_serve_gate (+ constant SERVE_ALIGN_MIN)
  still documents the ">0.1 serve-alignment floor" that WP1 REMOVED from
  _make_arc. It is a tautology (assertLessEqual(0.05, SERVE_ALIGN_MIN)),
  calls NO production code, and is GREEN, so I left it (out of INS-1-001
  scope; deleting a passing unflagged test = over-reach). But it now proves
  teeth of a gate that no longer exists -> stale; Test Dev should retire or
  re-point it to the new exact-zero tripwire semantics.

- WP1 (F041) DONE in surrogate_training.py _make_arc: guard
  `if abs(dot) <= 0.1:` -> `if dot == 0.0:` (exact-zero tripwire; sign
  now taken from first non-LensDomainError fraction, frac=0.5 nominal).
  Rewrote comment above the orientation loop: SIGN of dot fixes served
  two-image side (~12 orders float64 margin, min|dot|=4.4e-3 over prior);
  |dot| = fold-opening transversality ~1.5*gamma, NOT cusp-proximity, so
  no magnitude filter (magnitude filter WAS the F041 regression, same
  category error as retired _PROBE_ETA); exact-zero only skips measure-
  zero tangency (sign undefined); fallback fractions only for
  LensDomainError skips; sign invariant across them (global fold-arc
  property). NO new constant, NO smaller threshold. fallback tuple, sign
  assignment, break, except, `if sign is None: return None` tail
  byte-unchanged. ast.parse OK. Did NOT touch _branch_speed_profile,
  _find_cusps, or any other symbol. Behavior change from HEAD: gamma
  bands that previously failed _make_arc when |dot| fell in (0,0.1] now
  build arcs (larger set of buildable bands; lens_amplification_surrogate
  schema unchanged). OWED->Test Dev: any fixture asserting _make_arc
  returns None due to small-|dot| (near-cusp step) no longer trips at 0.1;
  re-point if such fixtures exist.

- WP3 (surrogate_training.py) DONE: replaced finite-eta census probes with
  exact geometry in fold orientation + caustic inradius.
  (1) DELETED _probe_arc_side + _PROBE_ETA(=0.05) constant (both file-local,
  only caller was _make_arc). (2) _make_arc: same fallback-fraction loop
  (0.5,0.35,0.65,0.2,0.8) but per theta try fold_dir=geometry.
  fold_opening_direction(gamma,theta,branch=branch) & _,normal=_tube_normal
  (SAME serve normal _tube_source uses -> serve-consistent), dot=float(fold_dir
  @normal); require abs(dot)>0.1 (near-0 = too close to cusp, step on); sign=
  1 if dot>=0 else -1; break; except LensDomainError continue; None if no frac.
  image_count HARDCODED 4 (parity constant, Professor Q2; astroid interior AND
  saddle deltoid lobe at kappa=0 both = 4 real images). (3) NEW module helpers:
  _radial_slope(gamma,branch,theta)=y.y' (=(1/2)d|y|^2/dtheta) via critical_point.
  source + caustic_derivatives (mirror of _speed_slope, no FD). _branch_inradius_
  candidates: (a) closed-form |y| at refined _find_cusps thetas, (b) smooth
  interior minima = UPWARD zero-crossing of h=y.y' between adjacent in-domain
  samples with min endpoint speed>0.2*median (skip cusp dips), brentq-refined;
  wrap bracket (hi<=lo) skipped. _closed_form_inradius: parity==1 single periodic
  branch [0,2pi); saddle both branches over each lobe wedge (mirrors _caustic_
  points enumeration). (4) _caustic_inradius: winding/enclosure test on discrete
  cloud UNCHANGED (points<4 ->(0,False) guard kept; parity!=1 -> enc False), ONLY
  radii.min() replaced by _closed_form_inradius. No discrete-cloud min / FD /
  step-size const remains in the six targets (_tube_normal's internal dth FD is
  a reused pre-existing helper per plan, not a target).
  SMOKE (real import): parse+import OK; both symbols absent. astroid inradius ==
  gamma EXACTLY (0.10..0.95), discrete dmin biased high up to ~4e-4; saddle
  matches dmin to 6dp, enc False. astroid arcs 2 (sign -1, img 4); saddle arcs 6
  (img all 4, signs {1,-1}). New sign vs OLD-probe sign: MATCH for astroid all g
  and saddle g=1.4; MISMATCH saddle g=1.6 (old=+1/img2, new=-1/img4). Direct
  find_images at eta 0.05..1e-4 CONFIRMS new is correct: sign=-1 side has 4
  images, +1 side has 2 -> old finite-eta probe MISLABELED the small deltoid
  lobe (the exact census failure WP3 removes). New = physically correct.
  BEHAVIOR CHANGE from HEAD: saddle arcs near gamma~1.6 now report inward_sign
  flipped (+1->-1) and image_count 2->4; astroid caustic_inradius diagnostic
  drops ~1e-4 to exact gamma.
  OWED -> Test Dev / Inspector:
  * test_lensing_exterior_windows.py uses st._caustic_inradius in
    test_anisotropic_gain_admits_fat_direction_refuses_thin (line ~1805) and
    test_isotropic_inradius_admission_loses_the_anisotropic_gain (~2694). Both
    use RELATIONS (inradius-eta_max<0.60, 0.60/inradius>1), preserved by the
    slightly-smaller exact inradius (astroid ir==gamma_mid<0.60) -> expected GREEN,
    but numeric inradius value changed ~1e-4; re-verify.
  * Any saddle fold-arc test keyed to the OLD near-gamma~1.6 inward_sign/
    image_count (=+1/2) now sees -1/4 (correct). Re-point if such fixtures exist
    (test_lensing_ppgo_bandsplit / arc-count suites).
  I did NOT run test suites or edit tests (role boundary).


- WP2 (cusp LOCATION = analytic root of y'.y''=0) DONE in
  surrogate_training.py: added `from scipy.optimize import brentq` (after
  numpy). New module helpers before _find_cusps: `_speed_slope(gamma,branch,
  theta)` = float(y'[0]*y''[0]+y'[1]*y''[1]) via geometry.caustic_derivatives
  (single-source g, no finite diff); `_refine_cusp_angle(gamma,branch,lo,hi)`
  = brentq(lambda t:_speed_slope(...),lo,hi,xtol=4*eps). _find_cusps sig now
  keyword-only gamma,branch (after periodic). DELETED _CUSP_SPEED_REL_FRAC
  constant+comment; inlined `window_dip_frac=0.2` LOCAL — detection+window
  loop (index i, lo/hi dip walk, span, delta=max(min_hw,ws*0.5*span)) kept
  byte-identical, so delta_theta is provably identical (0.2==old const).
  Only theta changes: per detected i, bracket lo=max(thetas[i]-step,
  theta_min+_CUSP_BRACKET_EPS), hi=min(thetas[i]+step,theta_max-eps)
  (step=median diff; new constant _CUSP_BRACKET_EPS=1e-9 keeps brentq off
  diverging saddle wedge edge). TWIN GATE (Professor): accept root ONLY if
  g(lo)<0<g(hi) (upward crossing=minimum) AND caustic_speed(root)<1e-6*
  speed.max(); else fall back to thetas[i] (never invent). try/except
  geometry.LensDomainError->fallback. 4 call sites pass gamma/branch:
  _astroid_arcs & _cusp_source_angles -> gamma=gamma,branch=1; _saddle_arcs
  & _lobe_cusp_source_angles -> gamma=gamma,branch=branch inside branch loop
  (kept saddle width_safety/min_halfwidth kwargs).
  Smoke (real import): parse OK, _CUSP_SPEED_REL_FRAC absent; astroid gamma
  0.37 n=201 OFF-GRID -> 4 cusps at 0,pi/2,pi,3pi/2, root_err ~1e-16 vs
  nearest_sample_err ~1e-2 (relocation works), speed<1e-15; saddle gamma1.6
  cusp at 0.0 delta0.0919; _astroid_arcs 4 / _saddle_arcs 6 cusps run;
  _cusp_source_angles=[-pi/2,0,pi/2,pi]. Did NOT touch tests.
  OWED->Test Dev: any test invoking _find_cusps directly must now pass
  gamma=/branch= (signature changed, keyword-only) or hits TypeError; tests
  referencing st._CUSP_SPEED_REL_FRAC now AttributeError (retired); near-cusp
  window fixtures unaffected (delta byte-identical) but cusp-ANGLE fixtures
  that expected the sampled minimum now get the ~1e-10 analytic root.

- WP1 (surrogate_training.py, swap FD estimators->exact geometry) DONE:
  (1) _min_curvature_radius: deleted inlined 3-pt circumradius + area2<1e-30
  collinearity guard; now thetas=linspace(theta_lo,theta_hi,max(n//2,32))
  ENDPOINTS INCLUDED, r_min=min over (band[0],band[1]) of
  geometry.caustic_curvature_radius(gamma,thetas,branch=arc.branch) (vectorised
  over theta). inf for straight point = no constraint (replaces guard). Value is
  4.9-9.6% SMALLER than incumbent (F038 endpoint bias) BY DESIGN. (2)
  _branch_speed_profile: deleted np.gradient/np.roll central diff//step; per-theta
  loop appends float(geometry.caustic_speed(gamma,theta,branch=branch)) w/
  try/except geometry.LensDomainError skip (whole-array caustic_derivatives
  refuses if ANY theta off-wedge -> can't vectorise). Kept periodic-vs-linspace
  theta + `<4 pts -> good_theta,np.array([])` guard. (3) _InteriorAdmission: added
  frozen field `gammas: tuple` (row-aligned w/ radius_grid/caustic_clouds); admits()
  now loops `for gamma_i,radius_axis in zip(self.gammas,self.radius_grid)`, builds
  same probes (rho_outer*radii), refuses via
  geometry.nearest_caustic_point(gamma_i,0.0,[px,py],kappa=0.0).distance<eta_max,
  NO margin, try/except LensDomainError->return False. admits_exterior UNTOUCHED
  (byte-identical, still cloud-based; caustic_clouds field kept). (4)
  _interior_admission: passes gammas=tuple(float(g) for g in band_gammas). Deleted
  _CLOUD_MARGIN_FRAC constant + its docstring block entirely.
  Smoke (real import): constant absent; min_curv=0.155 finite; speed 64/64 finite
  periodic; saddle gamma1.5 drops 64->16 in-wedge both branches; gammas=(0.45,0.5,
  0.55); admits interior True, near-caustic False. parse+import OK.
  OWED -> Test Dev (I did NOT edit tests; they encode retired cloud+margin contract):
  * test_lensing_exterior_admission.py: MANY tests mock.patch.object(st,
    '_CLOUD_MARGIN_FRAC',...) / assert st._CLOUD_MARGIN_FRAC>= / margin-width vs slop
    / _cloud_nearest_over_band cloud reproduction of admits -> now AttributeError
    (constant gone) + admits is exact-distance no-margin. Whole margin-sizing/false-
    admit-closure test group must be re-pointed to the exact nearest_caustic_point
    contract (or retired). test_exterior_admitted_set_unchanged_under_margin premise
    (interior margin doesn't move exterior) is now vacuous.
  * test_lensing_ppgo_bandsplit.py & test_lensing_exterior_windows.py: use .admits()
    but NOT the constant; semantics (interior admitted / exterior refused) should
    hold under exact distance, but near-boundary fixtures may flip 4.9-9.6% (curv) /
    margin-width (interior) — Test Dev to re-verify, not a production defect.

- WP2 (fold_opening_direction) DONE: added public
  fold_opening_direction(gamma, theta, *, kappa=0.0, branch=1) to
  chang_refsdal/geometry.py immediately after WP1's
  caustic_curvature_radius (before the ghost section). Separate closed
  form (NOT the theta-derivative cascade): cp=critical_point(gamma,theta,
  kappa=kappa,branch=branch) -> x=cp.image, e=cp.soft_axis, r2=x@x,
  xe=x@e, D2=(4*xe*e + 2*x - 8*xe^2*x/r2)/r2^2, return D2/norm(D2). Only
  point-mass term of y=Ax-x/|x|^2 (Ax has zero 2nd deriv). soft_axis
  sign ambiguity harmless BY INSPECTION: D2 uses e only via xe^2
  (e->-e invariant) and 4*xe*e (xe & e both flip -> product invariant),
  so NO sign correction. Inherits critical_point's LensDomainError (no
  re-derived domain checks). NO finite diff / probe / image count.
  Smoke (real production import): gamma0.3,theta0.7 -> unit vec norm=1.0;
  parity-boundary (gamma0.5,kappa0.5 -> |gamma|==1-kappa) refused
  LensDomainError. parse+import OK. Did NOT touch consumers or tests.

- WP1 (F038) DONE: added caustic_derivatives/caustic_speed/
  caustic_curvature_radius to chang_refsdal/geometry.py after r_caustic
  (before GhostDomainError). Analytic closed-form cascade (NO finite
  diff): u,u',u'' -> r,r',r'' -> p_i=(lam-+gamma)-lam*u (lam factor
  present, the load-bearing fix over the brief's p=-u typo) ->
  y'=p'rT+pr'T+prT', y''=... . Verified the WHOLE cascade by hand
  (u',u'',r',r'' derivatives re-derived, all match spec) AND against an
  independent mpmath 40-dps numeric derivative of the curve definition:
  worst |y'| err 7e-15, |y''| err 1.8e-14 (<< ATOL 5e-13); astroid pin
  R_c/3g|sin2t| in [0.999,1.001] (<3e-3). Positive parity forces b=+1 and
  IGNORES branch (branch=-1 finite, no nan, no sqrt RuntimeWarning, byte-
  = branch=1); macro saddle honours branch, raises LensDomainError BY NAME
  off-wedge (disc<-1e-12, np.any for arrays) or u<=0, mirrors
  critical_point max(disc,0) clamp + lam<=0 + |gamma|==lam walls.
  Vectorised: returns np.array shaped (2,) scalar / (2,N) array. Wrappers
  delegate. Straight point -> inf (legit). parse+import OK.

- INS-2-001/002/003 RESOLVED (2nd bounce; pipeline forced Coder to edit
  the stale test fixtures after Test Dev didn't). Probed real production
  fns to pick verified wave/refusing re-points, then ran the 6 named tests
  GREEN + all 3 full files (96 passed/11 skip/1 xfail, no fail):
  * airy_fold `_LADDER_NODES` fold radius 0.14->0.06 (L=30<L_MAX stays
    WAVE, fold arm serves; verified fold-not-cusp + F_op order=0). +F028
    comment. Sibling ladder tests (_ladder_route mirror is is_saddle-only
    so pos-parity wave nodes still label 'fold') unaffected.
  * fast_path FOP_REFUSALS[-1] (63,0.9,0.2)->(63,0.3,0.2): hard-core WAVE
    (unresolved dmin=0, both arms decline) -> RAISE
    SchwingerCertificationError. Fixes refusal test + scalar flip witness
    (both get refused>0 from this entry). +F028 comment.
  * fast_path grid flip witness: FOP_GRID_SQRT_S is shared by 6 tests so
    could NOT drop 0.9; above ceiling ANY |y|~0.9 is unavoidably geometric
    (L=w*0.9>54, resolved). So added a THIRD dispatch branch per Inspector's
    explicit sanction: on-axis supra outcomes refused=2(ss0.3 b0)/arm=2(ss0.3
    b0.7)/geometric=4(ss0.9). Geometric branch is DISPATCH-parity only
    (node_value == geometric_amplification byte-exact, order 0) NOT an
    accuracy gate -- independent geometric-accuracy gate STILL OWED to Test
    Dev. +served_geometric>0 anti-vacuity.
  * operator sheared test y=[1.0,0.0]->[0.08,0.0] (hard-core wave refuse,
    RAISE SCE); kept [0.05,0.0]. +F028 docstring.
  Key routing facts (w=63, gamma=0.2): on-axis ss<=0.3 -> wave (dmin=0
  unresolved); ss=0.9 -> geometric. To keep an above-ceiling node WAVE:
  w*|y'|<48 OR w*delta_min<4. geometric-served nodes report order 0.

- OWED->Test Dev (INS-1-001/002/003, positive-parity-resolved-first):
  Inspector routed 3 TEST-file findings to Coder after WP1 gave
  _positive_parity_grid its select_branch geometric branch. CONFIRMED
  production is CORRECT (WP1 routes above-ceiling nodes through
  select_branch(w, delta_min, w*|y'|); WP2 saddle via inf-cancel leg) —
  NOT a production defect. Findings are stale fixtures encoding the OLD
  'every above-ceiling pos-parity node hits arm/refuses' contract. Did
  NOT edit the tests: selecting fixtures + flipping assertions that
  certify my own WP1 change is self-grading (role: Coder never authors
  gates for own code; Test Dev re-points). Precise work order:
  * test_lensing_airy_fold.py: _LADDER_NODES 'fold' entry (line ~1838:
    ('fold',500.0,0.14,_RAY_ANGLE,_GAMMA=0.3,0,0)) now routes geometric
    (delta_min~0.134, w*delta_min~67>=RHO_END=4, L=w*|y'|=70>L_MAX=48).
    Re-point to STAY 'wave'/fold-served: drop into wave regime via
    w*|y'|<48 (smaller |y|/gamma) OR unresolved w*delta_min<4, so the
    fold arm is still consulted. _UNIFORM_LADDER_NODES derives from it.
    Keeps test_fixed_priority_fold_tried_before_cusp +
    test_served_value_equals_labelled_rung_bitwise meaningful. Add F028
    docstring note.
  * test_lensing_fast_path.py (~1506): 3 tests —
    test_fop_refuses_uncertifiable_contractions needs a genuinely
    hard-core node (unresolved w*delta_min<4 AND both arms decline) so
    refused>0; test_fop_grid/scalar_schwinger_arm_flip_witness need
    above-ceiling fixtures that stay wave/refusing (w*|y'|<48 or
    w*delta_min<4), else update witness expectation to geometric.
  * test_lensing_operator.py OperatorOracleTestCase::
    test_sheared_host_above_ceiling_refuses_schwinger: pick a sheared
    above-ceiling fixture that is genuinely wave-routed + hard-core
    (unresolved, no arm certifies) so SchwingerCertificationError still
    fires, OR split into geometric-served vs still-refusing. Cite F028.
  SEPARATE OWED: no test yet exercises the NEW geometric serve on
  pos-parity above-ceiling nodes — Test Dev to ADD that positive gate.

- WP2 (operator.py _saddle_grid): replaced hand-rolled
  `w>ceiling and w*delta_min>=RHO_END` geometric test with
  `select_branch(w_node, delta_min, math.inf)=='geometric'`. Per
  Professor ruling passed cancellation_exp=inf so strongly_cancelling
  leg is vacuously true and ONLY the resolution leg is live ->
  algebraically byte-identical to old `resolved` (verified:
  select_branch(70,1,inf)=geometric, (70,0.001,inf)=wave). Restructured
  cascade: ceiling is now the ENCLOSING `if w_node>ceiling`, inner
  if/else on select_branch (geometric branch) vs arm/ceiling_refusers
  (unchanged), else=batch tail (unchanged). math already imported
  (line 170). Docstring + inline pre-pass comment updated: states inf
  cancellation exp -> only resolution leg live, PRESERVES w>60 AND
  resolved boundary EXACTLY (boundary did not move), and that a saddle
  geometric-onset gate (L>L_MAX accuracy leg) is OPEN/UNMEASURED (F028
  sweeps positive-parity only; ceiling exhaustion explains wave
  unavailability not geometric accuracy). Untouched: delta_min
  compute-once guard, batch tail, refusal reduction. parse+import OK.

- WP1 (Build re: F028): `_positive_parity_grid` above-ceiling nodes now
  route via `select_branch(w, delta_min, w*y_prime_norm)`. delta_min +
  macro_matrix guarded behind `np.any(w>W_CEILING)` (skip quartic below
  ceiling, accept #6); y_prime_norm=sqrt(y_scaled@y_scaled) reuses the
  norm already computed (== cancellation_exponent/w, no per-node
  _mass_sheet_map). 'geometric'->geometric_amplification(physical y),
  'wave'->existing _uniform_arm_value fold/cusp then named refusal. w<=ceiling
  untouched/byte-identical. Frame discipline: physical y/beta/matrix only.
  Smoke: below-ceiling finite; gamma0.9/w500 geometric serves finite;
  wave-branch refusals still raise SchwingerCertificationError.
- OWED (other roles, per brief): _saddle_grid still uses resolved-only rule
  (w>ceiling AND w*delta_min>=RHO_END, no L>L_MAX) — likely separate WP.
  SPEC serving-ladder + FINDINGS(F028 ~1% O(1) tail) + todo.d/completed.d/
  spec_changelog.d fragments NOT done here (Inspector/Librarian own those).
  Existing tests encode old 'every above-ceiling node hits the arm' contract
  (test_lensing_levers refusal tests + brief's blast-radius list) — Test Dev
  to re-point.
