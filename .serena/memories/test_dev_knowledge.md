# Test Dev Long-Term Knowledge

- Premise repair, not tolerance repair: fix fixtures to a case where the
  physical premise holds; keep the original as a companion test PREDICTING
  its nonzero offset via an independent closed form.
- If the WP under test never landed, write an honest contract suite with an
  @expectedFailure hasattr guard that goes RED when the API lands.
  @expectedFailure covers the test body NOT tearDown — bump anti-vacuity
  counters BEFORE the assertion.
- Oracle independence (F002): AST name-forbidding guards MUST walk
  ast.Name.id / ast.Attribute.attr (never a source-substring — a production
  symbol can be a substring of the oracle's own name); extend the guard for
  every new oracle/mutation helper; add a positive control (a tainted oracle
  calling the module-under-test flips it red). Pure-mpmath oracles for phase
  gates; regularize singular integrands with a DIFFERENT scheme than the code.
- Mocking/falsification kit: inject buggy/old variants by patching the MODULE
  GLOBAL the fn resolves; patch an except-branch's exception NAME in consumer
  globals so the real refusal escapes; reach untriggerable states via mock
  side_effect / SimpleNamespace fakes reusing real sub-objects; under numba
  patch the FULL .py_func chain (F010); a serve gate: patch its in_domain to
  lie + feed a fake result to prove the gate has teeth. "Gate RED" = refusal
  raised OR error>tol; test refusals at the production operating point.
- Detect a silent fast-path fallback by spying the fast method (call_count==0
  under assertRaises = short-circuited; not-called on a served config = fell
  back); assert fallback==direct bit-identically via float64 .tobytes().
  Byte-identity vs HEAD: exec `git show HEAD:file` into a module registered in
  sys.modules FIRST, compare lnL + fiducial nodes max|diff|=0, red-check via
  np.nextafter. For a module with numba @njit(cache=True) decorators, the HEAD
  copy MUST be a REAL temp .py file loaded via
  `importlib.util.spec_from_file_location` (register in sys.modules before
  exec) — numba needs a real file locator; exec-ing HEAD source into a
  synthetic module raises "no locator available".
- When a production-scale absolute tolerance (eps<1e-3, nat tiers) is
  UNREACHABLE in a minutes-scale fixture, gate on a BUDGET-INDEPENDENT
  relationship: for lnL-from-envelope error use dlnL <= AMP * eps_dense *
  |lnL_exact| (dlnL~eps*SNR^2, |lnL|~SNR^2 => ratio O(1), measured peak ~0.84)
  PAIRED with a monotone-refinement positive control witnessing eps->target as
  nodes increase. A fixed nat budget is the WRONG currency (error scales with
  SNR^2). Never widen a real production gate — keep it RED/xfail with a green
  converged control. lru_cache trained surrogates (one train/process).
- Gate each path at its OWN numerical floor (aggregate can pass while a
  component fails). Identity gates across DIFFERENT node grids floor at engine
  reproducibility ~1e-11 not eps — set gate orders below the LOO stop so it
  still certifies "algebra, not interpolation". _snap lattice anchors are NOT
  bit-exact (round(x/dx)*dx) — assertAlmostEqual + idempotence (key==key(snap)),
  never assertEqual. Probing internals via reduced outputs: prove the
  reproduction reduces bit-identically first.
- Timing: structural gates (speedup ratio, subdominance); absolute ms only
  arithmetic-derived. Stochastic QMC lnlike is NOT bit-repeatable — pin
  determinism/JSON round-trip at the deterministic SUB-layer
  (_get_dh_hh_timeshift) with assert_array_equal, never the stochastic top;
  hour-scale importance-sampling oracles are infeasible as a minutes gate.
- Conditional-vs-marginalized round-trip: a single plain draw sits ~25-30 nats
  ABOVE lnL_marg (extrinsic Occam), so the consistency gate is a LOWER bound
  (max >= lnL_marg - delta). Get in-support vectors under Fixed*Prior by
  sampling the unit cube until lnposterior is finite.
- Phase-loss: np.exp(1j*x) range-reduces accurately — float64 loss lives in
  the w*tau MULTIPLICATION; demos need irrational-scaled factors or synthetic
  inputs checked vs an independent oracle.
- cogwheel lensing gotchas: ChangRefsdalChannels needs a >=2-pt strictly-
  increasing positive w grid (no scalar fixtures); _lens_dic has beta as the
  4th positional (pass lens params by keyword). Mass-sheet twin lnL invariance
  needs a second time term t_geo_twin = t_c - dt_ms - xi*(t_min_B - t_min_A)/2pi
  (read t_min from a throwaway eval; don't assume dt_ref==-dt_ms). Unlensed-
  injection near-truth reference = LIGHTEST lens, source OFF the caustic centre
  (y=(0,0) is a caustic singularity -> -inf). Census: saddle(det<0) signed=-2,
  positive(det>0) signed=0.
- Neighbor-suite reds from drift: report, don't touch. Fully revert probe/
  mutation edits (verify by read-back + pattern search). Shell gate: plainest
  command shape (`python -m pytest <file> -q`) from the WORKTREE root; retry a
  bare denial once, a reasoned denial binds. Heavy lensing suites run together
  -> MemoryError; run one file at a time.
- For self-contained HEAD functions (no module-level state), AST-extract just
  the FunctionDef (`ast.get_source_segment`) and exec in a minimal namespace —
  cheaper than loading the whole HEAD module side-by-side when the function
  doesn't need it.
- To fixture near-singular/underflow behavior of a quantity LINEAR in its
  inputs, solve for a nullspace combination (e.g. s such that h1+s*h2 drives
  the target to ~0) rather than hand-tuning — produces genuine catastrophic
  cancellation with O(1) intermediates.
- Before using a config as a dispatch/ladder probe fixture, verify it's
  actually served by the TARGET arm and not preempted by a higher-priority
  arm — check which arm served bit-for-bit, don't assume from parameter
  ranges alone.
- Huge test files can break the built-in Edit/Read tools (size/token limits,
  "not read yet") — use Serena `replace_content` (relative path) and
  `read_file` ranges + `get_symbols_overview` instead of a full-file read.
  `tests/output/` can be hook-blocked from `list_dir`/`find_file` ("Path
  ignored") — verify generated artifacts (plots) via plain Python
  (`pathlib.glob`) in the conda env instead.
- SDK now caps inlined short-term memories at 24KB (tail-kept); earlier
  entries survive only in git history, not the prompt.
