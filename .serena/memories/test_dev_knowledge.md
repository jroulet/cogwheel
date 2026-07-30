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
  A function with LOCAL (call-time) imports needs patching at EACH consuming
  module's namespace, not just where the target is defined. RECURSION TRAP:
  when a test helper must call the REAL implementation from inside a test
  that ALSO mock.patches that same name elsewhere, capture the real
  reference (`_REAL_FN = module.fn`) at import time, before any patching,
  and always call that captured reference — calling the module attribute
  from inside the helper risks recursing into the patched version.
- Detect a silent fast-path fallback by spying the fast method (call_count==0
  under assertRaises = short-circuited; not-called on a served config = fell
  back); assert fallback==direct bit-identically via float64 .tobytes().
  Byte-identity vs HEAD: exec `git show HEAD:file` into a module registered in
  sys.modules FIRST, compare lnL + fiducial nodes max|diff|=0, red-check via
  np.nextafter. For a module with numba @njit(cache=True) decorators, the HEAD
  copy MUST be a REAL temp .py file loaded via
  `importlib.util.spec_from_file_location` — numba needs a real file locator.
  When the change under test is still UNCOMMITTED, `git worktree add /tmp/x
  HEAD` IS the pre-change baseline — the cheapest proof that a neighbor
  suite's red is pre-existing rather than caused by the WP.
- Freeze bit-identity fixtures as `float.hex()` STRINGS (exact round-trip) and
  rebuild the fixture's inputs FROM the stored hex, so the guard isolates the
  functions under test from upstream drift in whatever generated the config.
- When a production-scale absolute tolerance (eps<1e-3, nat tiers) is
  UNREACHABLE in a minutes-scale fixture, gate on a BUDGET-INDEPENDENT
  relationship: for lnL-from-envelope error use dlnL <= AMP * eps_dense *
  |lnL_exact| PAIRED with a monotone-refinement positive control witnessing
  eps->target as nodes increase. A fixed nat budget is the WRONG currency.
  Never widen a real production gate — keep it RED/xfail with a green
  converged control. lru_cache trained surrogates (one train/process).
- SUBTRACTIVE-TERM ACCEPTANCE (do-nothing control): the certifying oracle is
  the F-normalized residual vs the exact total — assert resid(WITH the term)
  <= resid(WITHOUT it) + tiny on every ADMITTED config (config-agnostic, no
  tolerance table). Reachable-red = patch the admission threshold to 0.0 to
  force-admit a genuinely refused config and assert the ratio exceeds 1.
- A train==serve (boolean decision equality) assertion is VACUOUS unless the
  suite also carries a foil config on which the RETIRED mechanism disagreed
  between the two grids — construct the near-threshold config explicitly.
- Never anchor a suite on a brief's named refuse/admit configs without
  measuring them first: named coordinates routinely miss the intended regime
  (the "near-cusp" configs at theta_c=15/85 deg did NOT refuse — cusps sit
  near the theta=0/90 axes, so refusal needed theta_c~0.3 deg). Record the
  measured quantity in the test, not the brief's number. Same applies to a
  brief's cited constant/root: derive the EXACT closed form yourself for a
  tight gate (a brief's truncated literal can miss by ~1e-6, well inside a
  tight tolerance); the brief's literal is fine only for a serve/refuse
  straddle check where the tolerance margin dwarfs the literal's own error.
- When constructing a fixture meant to reveal a correction term's effect,
  sweep the coordinate the term actually DEPENDS ON (e.g. angular for an
  angular-cosine term), not one it's insensitive to — a radial sweep can
  hide a break that an azimuthal sweep reveals (same node count either side).
- Gate each path at its OWN numerical floor (aggregate can pass while a
  component fails). Identity gates across DIFFERENT node grids floor at engine
  reproducibility ~1e-11 not eps. _snap lattice anchors are NOT bit-exact —
  assertAlmostEqual + idempotence, never assertEqual. Probing internals via
  reduced outputs: prove the reproduction reduces bit-identically first.
- Phase/frame tests: `np.unwrap(np.angle(ratio))` has an intercept defined
  only MOD 2pi (the principal branch can give -2pi where 0 is meant) — wrap
  the fitted intercept into (-pi, pi] before comparing to 0; the SLOPE is the
  unambiguous quantity (equals -t_min for a min-subtracted frame).
- Timing: structural gates (speedup ratio, subdominance); absolute ms only
  arithmetic-derived. Stochastic QMC lnlike is NOT bit-repeatable — pin
  determinism/JSON round-trip at the deterministic SUB-layer
  (_get_dh_hh_timeshift) with assert_array_equal, never the stochastic top.
- Conditional-vs-marginalized round-trip: a single plain draw sits ~25-30 nats
  ABOVE lnL_marg (extrinsic Occam), so the consistency gate is a LOWER bound.
  Get in-support vectors under Fixed*Prior by sampling the unit cube until
  lnposterior is finite.
- Phase-loss: np.exp(1j*x) range-reduces accurately — float64 loss lives in
  the w*tau MULTIPLICATION; demos need irrational-scaled factors or synthetic
  inputs checked vs an independent oracle.
- cogwheel lensing gotchas: ChangRefsdalChannels needs a >=2-pt strictly-
  increasing positive w grid (no scalar fixtures); _lens_dic has beta as the
  4th positional (pass lens params by keyword). Mass-sheet twin lnL invariance
  needs t_geo_twin = t_c - dt_ms - xi*(t_min_B - t_min_A)/2pi (read t_min from
  a throwaway eval). Unlensed-injection near-truth reference = LIGHTEST lens,
  source OFF the caustic centre (y=(0,0) -> -inf). Census: saddle(det<0)
  signed=-2, positive(det>0) signed=0. `ChangRefsdalChannels(w).reset()`
  mutates IN-PLACE and returns None (NOT chainable) — call `ch = Chang
  RefsdalChannels(w); ch.reset(); ch.evaluate(gamma=, y=, beta=, kappa=)` (or
  `ch.geometry_partition(...)`) as separate statements, never chained.
- Neighbor-suite reds from drift: report, don't touch. Fully revert probe/
  mutation edits (verify by read-back + pattern search). Shell gate: plainest
  command shape (`python -m pytest <file> -q`) from the WORKTREE root; retry a
  bare denial once, a reasoned denial binds. Heavy lensing suites run together
  -> MemoryError; run one file at a time.
- A test-only change (no production edits) cannot regress an unrelated
  slow/heavy suite that doesn't import the touched symbols — verify via a
  grep for zero imports/references, then skip running it.
- For self-contained HEAD functions (no module-level state), AST-extract just
  the FunctionDef (`ast.get_source_segment`) and exec in a minimal namespace.
- To fixture near-singular/underflow behavior of a quantity LINEAR in its
  inputs, solve for a nullspace combination rather than hand-tuning.
- Before using a config as a dispatch/ladder probe fixture, verify it's
  actually served by the TARGET arm and not preempted by a higher-priority
  arm — check which arm served bit-for-bit.
- TOOLING: huge test files break the built-in Edit/Read tools — use Serena
  `replace_content` (relative path) + `read_file` ranges +
  `get_symbols_overview`. `tests/output/` is hook-blocked from `list_dir`/
  `find_file` — verify generated plots via `pathlib.glob` in the conda env.
  The bash hook also blocks `cat >>` (the word "cat"): append via Serena
  `insert_at_line` (0-based; insert at line == linecount to append).
- SDK caps inlined short-term memories at 24KB (tail-kept); earlier entries
  survive only in git history, not the prompt.
- When production adds a new REQUIRED positional field to a serialized-
  artifact constructor, update every test helper that rebuilds/re-saves that
  artifact with ALL fields — otherwise load() raises KeyError before reaching
  the intended validation error, silently invalidating the premise.
- To unit-test a method refactored out of a free function into an instance
  method, bind the REAL methods onto a lightweight stateless probe class
  (class attrs) and call as instance methods — preserves `self` dispatch.
- When picking the worst-case band edge for a coverage/admission fixture,
  check which direction is actually worst: exterior admission truth-sets are
  worst at the band edge with the LARGEST caustic reach, interior at the
  SMALLEST — they can point opposite ways (Build 8h-b6).
- To isolate one gate's teeth when tightening a threshold also reshapes an
  upstream derived quantity, re-run the gate function directly on the SAME
  fixed tile/sample set with only the threshold changed, rather than
  regenerating the fixture.
- Content-stable digest for a saved .npz: raw-byte hashing is flaky (zip
  member timestamps aren't reproducible) — hash sha256 over the LOADED
  arrays sorted by (name,dtype,shape,tobytes) instead; guard the digest
  helper itself with a save-twice-matches test.
- Golden/history-free regression recipe: build the frozen fixture once via
  a THROWAWAY generator script, print frozen bits as `float.hex()` literals
  (or a content digest), bake the literals into the test, then DELETE the
  generator — no HEAD import, no self-oracle, immune to future drift.
- Before assuming interpolation/new code is the runtime cost, profile:
  a coarse test grid can be slow because each cell re-invokes an expensive
  shared primitive (e.g. a caustic-reach sweep) that has nothing to do with
  what's under test — reduce grid density (keep odd counts to preserve an
  exact symmetry point) rather than optimizing the wrong function.
- Guard-bypass pattern for legacy fixtures broken by a new production guard:
  mock.patch the guard for the legacy path, but pair it with a reachable-red
  test that calls the SAME entry point unpatched and asserts the guard fires
  for real; the guard's own dedicated teeth-test lives in its owning suite.
- Don't trust a brief's stated sign/direction for a fix — measure it. A
  brief can claim "move outward/harder" while the code's own inline comment
  and the measured values show the opposite; encode the invariant that
  matches MEASURED behavior (e.g. "never easier than before") and flag the
  brief's direction error separately, rather than encoding the brief's claim.
- TWO-STAGE oracle for a numerically-differentiated closed form: STAGE-1
  validate the TRANSCRIBED curve itself against the shipping function's own
  output (tight tolerance, e.g. <=1e-13) BEFORE STAGE-2 high-precision
  differentiation of that curve — a single-stage oracle that only
  differentiates can hide a wrong transcribed curve (a wrong sign/constant
  survived a full round this way, F038).
- Served-values insensitivity gate: perturb the analytic input by measured
  physical increments and assert the served output moves less than the
  accuracy bar; if the perturbation instead pushes the config out of the
  served region into refusal (no "served-but-over-bar" window exists), the
  gate's teeth become "still served (not None)", and value-sensitivity must
  be demonstrated separately via a deliberately-wrong-value fixture.
- mpmath dps cross-check discriminator: increasing oracle precision (e.g.
  40dps -> 60dps) should NOT necessarily shrink a correct implementation's
  residual once the residual is float64-limited (oracle precision >> float64)
  — a dps-INVARIANT residual confirms the closed form is right, not wrong;
  the pass/fail discriminator is residual MAGNITUDE against the mixed
  atol/rtol gate, never "shrinks with dps".
