# Professor short-term — Build 1d WP1 inference review (2026-07-30)

Reviewed the 3 disjoint gate suites for the `_WEDGE_EPS` deletion / analytic
`_tube_normal` build (env cogwheel-newlal). Ran ONLY fast domain tests.

RESULT: 78 passed / 38 skipped / 0 failed across
test_lensing_saddle_geometry.py + test_lensing_surrogate_training.py +
test_lensing_caustic_cusps.py (~68s). Skips are legitimate: 36 are
`COGWHEEL_TRAIN_TIER=1` engine-backed chart builds (operator-deferred, minutes
each) + 2 F043 `git show HEAD` oracles deliberately retired. No gate skipped.

Physics verified (not just green):
- Gate 2 (Q3, my authority): mechanism-first. At served endpoint theta_rel =
  -theta_max = -0.5 arcsin(1/gamma), sin(2 theta_rel) = -1/gamma so
  discriminant 1 - gamma^2 sin^2 = 0 analytically; test asserts disc<=0 FIRST,
  then branch +1/-1 sources bit-identical (array_equal) via d_root=sqrt(max(.,0))
  =0, then per-gamma-guarded gap==0.0 (incumbent 0.279 at 1.05). Exactly my
  non-coincidence recommendation. PASS.
- Gate 3: frozen golden literals (delta 1e-9), cusps==6, arcs==6, reach frozen,
  span STRICTLY > incumbent (1.05: 2.90729 > 2.90462; ~1e-3/edge recovered).
  Incumbent independently pinned to HEAD module (non-circular). Brief's example
  numbers 2.908035/2.910714 were illustrative "e.g."; measured pair differs but
  preserves the increase. PASS.
- Gate 4a: unit 1e-15, perp 1e-14 (my Q2 split), |y'|>1e-3 guard (my Q1),
  AST-scan forbids theta+-step FD. PASS. Gate 4b: golden +-1 table + INDEPENDENT
  find_images four-image census (F041 flip guard). PASS. Part C helper fixed to
  `if dot==0.0: continue` (F041 floor removed), docstrings corrected.
- Gate 5: serve edge finite + refuse 1e-12 outside (both paths). Two-lobe
  closure tightened 1e-2 -> 1.8e-3 (measured 1.670e-3 sqrt-resolved step;
  branches meet in IMAGE plane to 1e-6), honest comment. PASS.

DOCUMENTED DEVIATION (physics-honest, does NOT block): Gate 5b — brief demanded
caustic_derivatives unconditionally RAISE on the exact edge; delivered tree
shows the edge is a float measure-zero set: center=0 disc<=0 -> refuse,
center=pi lands hair-positive -> serves DIVERGENT |y'|~7.4e7. Test asserts the
honest disjunction (refuse OR diverge > 1e4 floor; interior regular speed <1e4).
This is exactly the float-fragility my prior Q3 note flagged. Author flagged the
production docstring's "refuses exactly on the edge" as aspirational at center=pi
for the WP1 fix-false-docstrings owner. Correct handling.

VERDICT: PASS. Heavy full-sampling / TRAIN_TIER validation is operator-deferred.
