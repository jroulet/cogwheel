# Test Dev Short-Term Observations

## 2026-08-21 (low_w_chart fixture-comment re-derivation — rename+teeth specs were ALREADY done)

- BOTH of my assigned specs were already fully implemented by the earlier
  shard; the only genuinely-remaining work was re-deriving two STALE
  fixture-constraint COMMENTS (the "rename fallout" grep is clean and the
  teeth test `CuspFrefNonVanishingSelfFalsificationTestCase` already pins
  the rho=2.0 guard decline). Verify-elsewhere before authoring: run the
  suite + grep before assuming a spec needs new tests.
- MEASURED (for the next surface move): gp=0.8 on-caustic rho=1.0 is 16/16
  buildable over the exact-chart gp x theta grid; exterior rho=2.0 is
  buildable in ~42% of theta (fold band [0.6,1.0] + Pearcey-fallback cusp
  band [0.35,0.6]), declined in cusp-RESOLVED dirs (theta~0 and >~1.0) by
  the non-vanishing guard (P->0), NOT by an absent form. rho=1.1 declines
  2/16 cells (gp 0.8/0.9 at theta=1.4), down from the pre-fallback 6/16.
- CONFIRMED the guard declines ONLY at rho>=1.1 (exterior); interior rho in
  [0.6,1.0] has essentially NO declined cusp cells, so the spec's "interior
  cusp cell" teeth fixture is unreachable and rho=2.0 is the correct
  substitution (matches test_dev_short_term's earlier finding).
- PRODUCTION FLAG (test-side can't fix): `fold_cusp_reference` emits
  RuntimeWarning "invalid value encountered in scalar divide" when the
  Pearcey reference is ALL-ZEROS (cusp cluster fully resolves -> uniform==0
  for every node) because `ratio = magnitude.min()/magnitude.max()` is 0/0
  before the `isfinite` guard catches it. Guard still declines (correct),
  but noisy -- hand to Coder to guard `magnitude.max()==0` first.

## 2026-08-21 (low_w_chart cusp-fallback test suite)

- CUPS-CELL PEARCEY FALLBACK IS RHO-LIMITED (key test-design finding): the
  fold->cusp b3->0 transition at gp=0.8 is cusp-REFERENCE-BUILDABLE only near
  the caustic. At rho=1.2, theta in [0.15,0.2] the Pearcey fallback
  (`cusp_uniform_reference` = cluster_sum * (P/P_asymp)) is NON-vanishing
  (min/max ~0.25); at rho>=1.5 the SAME b3->0 window gives matched->0 above
  w~7 (cusp cluster fully resolves, cluster_sum->0) so min|F_ref|==0 exactly
  and the `_NON_VANISHING_MIN_RATIO=1e-3` guard DECLINES the cell (None).
  The Architect's specs pinned rho=2.0 as "a genuine cusp cell" -- it is
  declined, not non-vanishing. Substituted rho=1.2 for the non-vanishing /
  node-exact / continuity specs and pinned the rho=2.0 decline as the guard's
  self-falsification (teeth). This is the cusp analogue of the fold grid cap
  at rho=1.05: a "cusp cell" fixture must be chosen where the Pearcey form
  actually holds, not at the build-brief's b3-measurement point.
- `_build_exact_cusp_chart` grid (gp [0.6,0.7,0.8,0.9] x rho [0.8,1.0,1.2,1.4]
  x theta [0.15,0.2,0.3,0.4] x _W_GRID) has every cell F_ref-buildable and
  puts the cusp cell (0.8,1.2,0.2) interior; node-exact serve rel err ~1e-14.
- fold/cusp handoff |F_ref| ratio at rho=1.2 peaks 3.12x (w~0.65), NOT the
  naive ~2.3x from a few sampled w -- sweep the FULL w-grid before choosing
  the continuity bar (set 5.0).
- serena insert_before_symbol on a module constant can land MID-BLOCK: it
  split the `_CENSUS_W_GRID` `#:` doc-comment from its variable when the
  target constant's own comment block precedes it. Verify surrounding
  structure after every such insert.
