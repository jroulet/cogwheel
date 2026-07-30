# Inspector Short-Term Observations

## 2026-07-29 — Build arc_guard_fix RE-REVIEW (same tree, gate re-invoke) — PASS
Re-verified the exact build my prior entry passed; redid checkable invariants
live (did NOT trust prior pass):
- Guard fix intact: _make_arc takes sign from
  geometry.fold_opening_direction(gamma,theta,branch) @ _tube_normal normal;
  skips only exact dot==0.0; fallback fractions step past LensDomainError;
  image_count=4 constant. abs(dot)<=0.1 GONE.
- Geometry callees exist w/ matching sigs in
  cogwheel/lensing/chang_refsdal/geometry.py (NOT cogwheel/lensing/geometry.py):
  fold_opening_direction/caustic_speed/caustic_derivatives/
  caustic_curvature_radius/nearest_caustic_point, all (gamma,theta,*,kappa,branch).
- Only sig change = _find_cusps (added REQUIRED kw-only gamma,branch). Swept
  every caller: prod 597,635,1473,1599,2060 pass gamma+branch; tests
  surrogate.py:1068 (ex-stale INS-1b-002) now gamma=gamma,branch=1;
  surrogate_training.py 1006,1660,1662 pass them; 1948 mock forwards **kwargs;
  caustic_cusps st._find_cusps 477,525,980,1005 pass them; _head_find_cusps AST
  oracle keeps OLD sig on purpose (independent HEAD reconstruction).
- _InteriorAdmission gained REQUIRED `gammas`; only ONE constructor (1823)
  passes it; no test builds it directly. Safe.
- F041 acceptance asserts VALUES through real stable_gamma_bands/
  detect_caustic_structure; self-falsification mock-strips small-gamma arcs.
- RAN live: import OK; pytest -k F041 = 5 passed; cusp subset of surrogate.py
  (incl ex-stale caller) = 4 passed. Did NOT re-run full 222/220/63s suites —
  prior entry ran them green on identical tree; proportionate.
- INS-1-002 STILL OPEN: SPEC.md row 55 unchanged (git diff --stat empty), still
  narrates retired estimators. Pre-existing (crashed 1b), Librarian doc-sync per
  brief — flag only, NOT this build's defect, NOT resolved.


## 2026-07-29 — Build arc_guard_fix REVIEW (guard fix + caller fixes + acceptance) — PASS

Scope: uncommitted worktree /home/tejaswi/Work/cogwheel-claude-dev. Diff-vs-HEAD
looks huge (surrogate_training.py 427 lines) because HEAD predates crashed
Build 1b (retire six numerical estimators), which is INHERITED + driver-verified
per brief_arc_guard_fix.md. THIS build's real change = ONE production edit
(_make_arc guard) + test-caller fixes + one acceptance test.

### WP1 guard fix CORRECT (brief option 1: delete the magnitude guard)
- _make_arc (surrogate_training.py:659-711): sign from
  geometry.fold_opening_direction(gamma,theta,branch) dotted with the serve
  normal (_tube_normal); skips only exact dot==0.0 (measure-zero tangency);
  fallback-fraction loop (0.5,0.35,0.65,0.2,0.8) steps past LensDomainError;
  image_count=4 parity constant. Old `abs(dot)<=0.1` GONE. Signature unchanged
  -> _make_arc callers unaffected.

### Checklist 5b swept: only ONE signature change in whole build = _find_cusps
  (added REQUIRED kw-only gamma,branch). Verified EVERY live real caller passes
  them: prod surrogate_training.py 598,636,1474,1600,2061; tests
  surrogate.py:1069, surrogate_training.py:1007,1661,1663,
  caustic_cusps.py:478,526,981,1006. The _head_find_cusps/head_find_cusps AST
  oracle (caustic_cusps.py:269-305,1008) DELIBERATELY uses OLD signature — it
  reconstructs HEAD's self-contained fn as an independent oracle; NOT a stale
  caller. test_lensing_exterior_windows.py uses only signature-STABLE
  _caustic_inradius (unpacked correctly), no _find_cusps -> no break.

### Removed constants (_PROBE_ETA,_CUSP_SPEED_REL_FRAC,_CLOUD_MARGIN_FRAC,
  _probe_arc_side) grep-clean in prod. In caustic_cusps.py they survive only as
  (a) the self-contained HEAD-oracle exec namespace and (b) local
  INCUMBENT_CLOUD_MARGIN_FRAC=0.10 test constant — that suite ran green.

### Acceptance test CORRECT (surrogate_training test file, ran & passed)
- StableGammaBandsF041TestCase (3 methods = brief acceptance 1/2/3) +
  StableGammaBandsF041SelfFalsificationTestCase (2 methods, mock.patch strips
  small-gamma arcs -> dropped!=[] AND zero-arc band; positive controls).
  Asserts VALUES: dropped==[], len(arcs)>0, image_count==4, inward_sign in
  {-1,1}, cross-gamma label/count stability. APIs verified: stable_gamma_bands
  -> (list[((lo,hi),CausticStructure)], list[(lo,hi)]);
  detect_caustic_structure(gamma,parity,*,n_samples=); CausticStructure is
  @dataclass(frozen=True) with `arcs` field -> dataclasses.replace(...,arcs=())
  valid; FoldArc has inward_sign/image_count. imports dataclasses(168),
  mock(181) present. _CountingTestCase anti-vacuity honored.

### All four affected suites RUN GREEN (not merely collect)
- test_lensing_surrogate_training.py: 9 passed, 36 skipped (4.6s); 5 F041 tests
  confirmed RAN via -v (not skipped).
- test_lensing_surrogate.py: 51 passed, 1 skipped (222s).
- test_lensing_exterior_admission.py: 38 passed (220s).
- test_lensing_caustic_cusps.py: 25 passed (63s).

### INS-1-002 STILL OPEN (Librarian doc-sync, flag-only, NOT this build's job)
- SPEC.md unchanged this build. Row 56 still narrates the RETIRED mechanisms:
  "fold arcs probed at multiple interior thetas (near-axial F012 refusals...)"
  = the deleted _probe_arc_side census probe (now fold_opening_direction
  geometry); "cusp locations derived from caustic-speed minima" (now analytic
  root of y'.y''=0). NOTE: SPEC row 54 (geometry.py engine) ALREADY lists
  caustic_derivatives/caustic_speed/caustic_curvature_radius/
  fold_opening_direction correctly — those pre-exist HEAD. COVERAGE_DESIGN.md
  also still describes retired estimators. Per brief + ownership split: flag to
  Librarian, do NOT edit canonical surfaces in a WP. Carried forward, NOT
  resolved, NOT a build defect.

## Pattern reinforced
- When diff-vs-HEAD dwarfs the plan, check whether HEAD predates an INHERITED
  uncommitted crashed build; read the handoff brief's DONE/UNDONE inventory
  before treating the big diff as scope creep. The real WP was tiny.
- A behavior-fix rung removing a pathology lands clean only if the sibling
  suite that pinned the pathology was rewritten; confirm removed test/constant
  names grep-empty in prod AND RUN each affected file green (kw-only-arg
  breakage COLLECTS clean, fails at run).
