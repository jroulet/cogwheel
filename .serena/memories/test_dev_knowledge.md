# Test Dev Long-Term Knowledge

- Premise repair, not tolerance repair: fix fixtures to a case where the
  physical premise holds; keep the original as a companion test PREDICTING
  its nonzero offset via an independent closed form.
- If the WP under test never landed, write an honest contract suite with
  an @expectedFailure hasattr guard that goes RED when the API lands.
- Inject buggy/old variants by mock.patching the MODULE GLOBAL the
  function resolves; align test-side reproductions with drifted
  production arity; neighbor-suite reds from drift: report, don't touch.
- Extend AST/name-forbidding guards for every new mutation/oracle helper
  (F002); pure-mpmath oracles for phase gates. For rules differing only
  in edge cases, assert a sub-case where old and new agree bit-for-bit.
- Fully revert probe/mutation edits; verify by read-back + pattern search.
- Shell gate: plainest command shape (`python -m pytest <file> -q`), from
  the WORKTREE root; retry a bare denial once; a reasoned denial binds.
- Falsification under numba: patch the FULL .py_func chain (F010); "gate
  RED" = refusal raised OR error > tol; test refusals at the production
  operating point, not the accuracy-study setting.
- A plan-anticipated gate exposing a production shortfall stays RED/xfail
  (no tolerance widening) paired with a green converged positive control.
- Gate each path/component at its own numerical floor; an aggregate gate
  can pass while a component fails — keep both. Timing: structural gates
  (speedup ratio, subdominance); absolute ms only arithmetic-derived.
- np.exp(1j*x) range-reduces accurately — float64 phase loss lives in the
  w*tau MULTIPLICATION; phase-loss demos need irrational-scaled factors.
  If a gate's claimed band is unreachable from realistic fixtures, build
  SYNTHETIC inputs checked against an independent oracle.
- Probing internals exposed only through reduced outputs: prove your
  reproduction reduces bit-identically to production first.
- ChangRefsdalChannels needs a >=2-point strictly-increasing positive w
  grid (no scalar fixtures); _lens_dic has beta as 4th positional — pass
  lens params by keyword.
- Lattice anchors from _snap are NOT bit-exact (round(x/dx)*dx): use
  assertAlmostEqual + snapping-idempotence (key == key(snapped)), never
  assertEqual; the 1-ULP offset floors ratio quantities ~1e-16.
- Identity gates across DIFFERENT node grids floor at ~1e-11 (engine
  reproducibility), not eps — set the gate orders below the LOO stop so
  it still certifies "algebra, not interpolation".
- Detect a silent fallback by wrapping the fast-path method (not-called
  => fell back); assert fallback == direct bit-identically via float64
  .tobytes(). Reach unreachable guard states with types.SimpleNamespace
  fakes reusing real sub-objects so earlier guards still pass; refusal
  fallbacks via mock.patch side_effect on the cache/fiducial builder.
- @expectedFailure covers the test body, NOT tearDown — bump anti-vacuity
  counters BEFORE the assertion.
- Mutation for except-branches: patch the exception NAME in the consumer
  module's globals to an unrelated type so the real refusal escapes
  (gate RED); untriggerable branches via mock side_effect.
- Mass-sheet twin lnL invariance needs a second time term from per-config
  t_min referencing: t_geo_twin = t_c - dt_ms - xi*(t_min_B-t_min_A)/2pi;
  read t_min from a throwaway ChangRefsdalChannels eval, don't bet on
  dt_ref == -dt_ms.
- Unlensed-injection near-truth reference: LIGHTEST lens with source OFF
  the caustic centre — y=(0,0) is a caustic singularity -> -inf.
