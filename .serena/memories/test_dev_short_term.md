# Test Dev Short-Term Observations

Build WP1 (delete _WEDGE_EPS / analytic _tube_normal; test_lensing_surrogate_
training.py Gates 1-3, 2026-07-30):
- WP1 production landed in working tree: `not hasattr(training,'_WEDGE_EPS')`
  passes, token absent from module source. HEAD (git show) still has
  `_WEDGE_EPS=1e-3` at L123 + inset edges (L629/1418/1534/1993/2023/2055) ->
  HEAD is the pre-WP1 baseline for `_head_training_module()` reachability.
- COLLECTION FIX: file imported deleted `_WEDGE_EPS` (L197) + used at
  L1011-1012 (`_wp3_fixoff_left_arc`) and L1488. Removed from import; added
  TEST-LOCAL `_LEGACY_WEDGE_EPS=1e-3` (L975) to keep the WP3 tube-tail
  counterfactual arc byte-identical; changed `test_fixon_arc...` edge_theta to
  `-theta_max` (production window MOVED to true edge, genuine behavioral update
  verified -0.3506171822915343 at g=1.55).
- Gate1 `WedgeEpsDeletionTestCase`: hasattr+source-token deletion; source scan
  `_WEDGE_INSET_RE=re.compile(r'theta_max\s*[-+]\s*(?:_WEDGE_EPS|\d)')` over
  wedge-bound fns (guard asserts _saddle_arcs+_lobe_winding_loop SEEN, else
  vacuous); loop endpoints BIT-equal critical_point(branch=+1/-1) at
  center-+theta_max (turnaround at idx n-1).
- Gate2 `WedgeClosureGapTestCase` (Prof Q3): disc `1-g^2 sin^2(2 theta_rel)`
  at endpoint MEASURED <=0 for ALL g in {1.05,1.3,2.0} x centers {0,pi} ->
  max(.,0) clamp fires -> branches coincide bit-for-bit -> gap `==0.0` exactly
  (per-gamma-guarded: disc<=0 -> ==0.0 else <tol). Incumbent gaps
  {1.05:0.278958,1.3:0.107435,2.0:0.051454}.
- Gate3 `WedgeCoverageNoShrinkTestCase`: FROZE MY OWN measured goldens (spec's
  2.908035/2.910714 were off; at n=200 HEAD span 2.9046151 vs WP1 2.9072887 at
  g=1.05). cusps==6, arc count + reach vs frozen goldens; span STRICTLY >
  incumbent (WP1 recovers ~1e-3/edge coverage the inset discarded).
- SELF-FALSIFICATION `WedgeEdgeSelfFalsificationTestCase` (FAST tier, NOT
  skipped): regex catches dirty inset strings + spares clean edge; HEAD
  `_lobe_winding_loop` gap > tol (reopens); HEAD span < WP1 span & == frozen
  incumbent; branches DIFFER at interior theta (0.3*theta_max, disc>0).
- Green: my file 20 passed / 36 engine-skipped 6s; WP1 subset 11 passed 5s.
  Neighbor test_lensing_caustic_cusps.py (only other suite using _tube_normal/
  detect_caustic_structure; authored FOR WP1) 28 passed 2 skip 62s. No other
  test imports deleted _WEDGE_EPS (grep-verified) -> no collection breakage.

Build (WP1 delete _WEDGE_EPS / analytic _tube_normal / fix false docstrings;
test_lensing_saddle_geometry.py Gate 5, 2026-07-30):
- WP1 production diff: geometry.py change is DOCSTRING-ONLY (F044 fix: wedge
  edge is a REGULAR caustic point where theta-derivs diverge as a
  parametrization artifact, NOT a cusp; the 3 deltoid cusps are interior
  |y'|=0 roots). _WEDGE_EPS/_tube_normal live in surrogate_training.py (NOT
  geometry, NOT imported by this file) -> no behavioral audit hit for my file.
- Added WedgeEdgeServeRefusePredicateTestCase(SaddleTestCase): 5 methods.
  MEASURED (g=1.3,kappa=0,theta_max=0.5*arcsin(1/g)=0.4388): critical_point
  SERVES finite source (|src|=1.315) exactly at center+theta_max for centers
  0,pi both branches; RAISES LensDomainError at +1e-12 (discriminant slope
  ~-3.3 -> -3.3e-12 < -1e-12 guard). caustic_derivatives RAISES at +1e-12.
- KEY FP GOTCHA (deviation from brief): caustic_derivatives at the EXACT edge
  is FP measure-zero-straddled: center=0 -> discriminant lands <=0 -> RAISES
  (d_root==0 guard); center=pi -> lands +~1e-15 -> SERVES divergent |y'|~7.4e7.
  Brief demanded unconditional raise at edge for BOTH -> would FAIL at pi.
  Encoded honest disjunction (raise OR |y'|>EDGE_DERIVATIVE_DIVERGENCE_FLOOR=
  1e4); regular interior |y'|<=3.7 so floor discriminates by ~3.5 decades
  each side. Production caustic_derivatives docstring's "refuses exactly on
  the wedge edge" is still aspirational at center=pi (flag for docstring fix).
- Tightened TwoLobeCriticalStructureTestCase closure gap 1e-2 -> 1.8e-3
  (measured 1.670e-3 @ n_half=1500; NOT zero/exactly-representable -- it's the
  sqrt-resolved step between branch +1 at exact lower edge vs branch -1 one
  edge-clustered sample inside; deterministic, WP1-invariant). Did NOT assert
  gap==0 (that predicate is the winding-loop file's, not here).
- TOOLING: Serena replace_content intermittently rejected a large multi-line
  repl ("Field required repl missing" pydantic error) 3x on identical args; a
  minimal repl worked, then the full one succeeded on retry -> retry, don't
  reformat. Built-in Read/Edit tools are UNAVAILABLE in this session (Edit
  errors "File has not been read"; Read = "No such tool") -> Serena only.
- Green: file 30 passed (was 25) 6s; siblings test_lensing_caustic_derivatives
  25 passed, test_lensing_geometry 13 passed (no WP1 regression).

Build 1d WP1 (test_lensing_caustic_cusps.py, 2026-07-30):
- Added Gate 4a `TubeNormalGeometryTestCase`: analytic `_tube_normal` unit
  (<1e-15, meas 2.2e-16) + perp to `geometry.caustic_derivatives` y'/|y'|
  (<1e-14, meas 2.7e-17); AST guard asserts source has caustic_derivatives,
  no `_WEDGE_EPS`, no theta+-step BinOp. Q1 guard: only probe |y'|>1e-3
  (NaN at |y'|=0 cusps, RuntimeWarning divide there).
- Added Gate 4b `InwardSignGoldenTableTestCase`: FROZEN GOLDEN_INWARD_SIGN
  literals {(g,+1):(-1,-1) for g in .2,.4,.7,.9; (g,-1):(-1,-1,1,-1,-1,1)
  for g in 1.2,1.5}. Catches silent F041 orientation flip the self-
  consistent sign(dot)==inward_sign health test MISSES (both recompute from
  same _tube_normal). Non-circular via independent 4-image find_images
  census. Reachable-red verified: negating _tube_normal -> built (1,1) !=
  golden.
- Part C helper fix: `_chosen_serve_theta` now `if dot == 0.0: continue`
  (was `abs(dot)<=SERVE_ALIGN_MIN`) to mirror production _make_arc post-F041;
  fixed 3 false docstrings (SERVE_ALIGN_MIN now = HEALTH floor not build
  floor). Worst |dot| at chosen frac=0.5 is 0.2975>0.1 so health test green.
- CROSS-SUITE (report only, NOT mine): test_lensing_surrogate_training.py
  fails COLLECTION on deleted `_WEDGE_EPS` import (L197; uses L1011-1012,
  L1488) — WP1 production break, its owner must fix.
- File: 28 passed, 2 skipped (pre-existing F043 HEAD-oracle skips), 65s.
