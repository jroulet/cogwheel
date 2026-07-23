# Coder Long-Term Knowledge

- "Coders write, downstream verifies": if the sandbox denies runtime
  checks, verify what's checkable read-only, list UNVERIFIED plainly;
  don't retry denied calls.
- Verify a plan's "code-pinned"/"already exists" claim by grep/find_symbol
  before building on it; if absent and supplying it needs out-of-scope
  design, BLOCK and escalate — never fabricate an oracle (Build 3e).
- Never author certification gates for your own WP code; when retiring
  superseded tests, leave a loud OWED list naming replacement gates for
  Test Dev.
- After a regex-anchored method retirement, re-parse the file (ast.parse):
  substring anchors can leave a silent IndentationError.
- Cubic splines are C2: put a NODE on each C2 kink, never a segment break;
  interpolate only the single smooth object (demodulated envelope) and
  rebuild analytic/switched parts closed-form at dense samples.
- LOO adaptive refinement: held-out error from a few-nearest-OTHER-node
  fit, normalized in the gate's error currency, hard-coded threshold.
  Keep ONE shared refinement loop parameterized by a node_error closure
  (DRY across direct/ratio paths).
- Include the known worst-case point in every seed grid so refusals fire
  unswallowed on first eval; fallback paths must forward the already-
  computed seed — never redo engine work on fallback.
- Don't catch/fallback across independently gated branches: try/except
  ONLY around the cache/fiducial build; candidate-side refusals propagate
  (refusal symmetry with the exact path).
- except clauses name refusal types specifically (LensDomainError IS-A
  ValueError, CancellationError IS-A RuntimeError) so unrelated errors
  still propagate.
- cogwheel Prior convention: ln_jacobian_determinant returns
  log|d(sampled)/d(standard)| (INVERSE-transform Jacobian), args =
  standard_params+conditioned_on — verify signs against gw_prior/mass.py
  and extrinsic.py templates, never trust WP-text signs. Prior MRO: mixins
  precede Prior (check_inheritance_order); IdentityTransformMixin already
  includes UnitJacobianMixin; in CombinedPrior.prior_classes a
  conditioned_on provider must come earlier.
- JSONMixin serializes via get_init_dict, NOT __dict__; pickle uses
  __dict__ — __getstate__ drops derived caches, __setstate__ rebuilds them
  empty, behavioral flags (testing seams) PRESERVED. For an optional
  not-yet-serializable feature: override get_init_dict to POP the key when
  default (JSON byte-identical vs HEAD) and raise NotImplementedError when
  set (defer fitted serialization); the object rides pickle in __dict__.
- Numerical series: prefer reciprocal-binomial O(1)-scaled factorization;
  large phases lose precision in the w*tau MULTIPLICATION — reduce mod
  2*pi first. Accuracy tests near singularities need scale-aware bounds
  (~eps * summed-term magnitude) plus a canary for the flat gate.
- Before reusing a "shared" primitive, check the call site for redundant
  re-derivation. A flat parameter-independent "floor" can be an exact
  closed-form limit — verify before short-circuiting. Filtered "real-only"
  comparison sets can miss the nearest virtual member — check the spec.
- numba: njit freezes module globals/callees (test via full .py_func
  chain, F010); fastmath=False is load-bearing where error-free transforms
  exist; njit only pure float64/complex128 loops; explicit loops change
  accumulation order — re-certify in deep-cancellation regimes; scalar API
  delegates to the batched core (gather/one call/scatter, guard empties).
- Worktree: built-in Edit fails on main-tree absolute paths — use Serena
  replace_content with relative paths from the worktree.
- Prove an untouched path stays byte-identical after additive edits: load
  the HEAD module copy via importlib SIDE-BY-SIDE (register it in
  sys.modules FIRST so @dataclass fields resolve) and assert max|diff|=0.0
  over a config sweep + full refusal-decision match. Make the new-regime
  classification gate EXACTLY mirror the frozen path's gate so the new
  prefix returns before touching frozen internals (byte-identity by
  construction). A SINGLE fast-path intercept at the top of the expensive
  method that returns None on every guard miss lets the exact path fall
  through untouched when the feature is off or any guard fails.
- Tighten a tolerance ONLY in a sub-region without touching the certified
  hot path: gate the constant on a PURE fn of the candidate params at the
  single shared decision site (split _LOO_STOP -> FAST/STRONG keyed on
  gamma'); the unchanged branch returns the OLD constant verbatim so the
  certified region stays byte-identical and fiducial/cache purity holds.
  Key on the PHYSICALLY correct variable (gamma', not |gamma|).
- Independent oracles for singular integrands must be regularized: a naive
  Int_0^inf t^{s-1} h(t) form is ill-posed — use subtract-h(0) or IBP-with-
  h'. A DIFFERENT regularization scheme from the code's is the point (F002
  non-circular); phase agreement also confirms sign/conjugation convention.
- Serena `replace_symbol_body` on a function target: the new body MUST
  include the `def` signature line — omitting it deletes the def+docstring,
  producing a column-0 IndentationError.
- Never hand-tune a calibration/curvature constant just to make your own
  certificate pass — that fits the oracle to the code under test
  (self-grading); leave it refusing and flag the calibration as owed work.
- For large Monte-Carlo census sweeps (N>=1e5), stream results into fixed
  threshold-grid histograms (`counts_ge += (arg>=grid).sum()`) instead of
  storing per-sample arrays — the histogram IS the CDF, no memory blowup.
- To classify which guard blocks a sample, toggle ONE guard off via
  `dataclasses.replace` on a frozen config object and re-call the real
  guard function — never re-derive the guard math inline.
- SDK now caps inlined short-term memories at 24KB (tail-kept); earlier
  entries survive only in git history, not the prompt.
- Prefer OPTIONAL trailing args with backward-compatible defaults over
  changing a function's return-tuple shape when adding capability — keeps
  existing call sites/tests passing as regression guards for the untouched
  path (Build 8h-b).
- A shared derivation helper (DRY refactor) dedupes CODE but not
  necessarily RUNTIME: if several call sites each independently invoke the
  shared helper, an expensive underlying computation (e.g. a full geometry
  sweep) can still run once PER CALLER — for genuinely expensive shared
  derivations, compute once and pass the result to all consumers.
- When a boundary/refusal is known to be monotone over an ordered grid,
  bisect on the node INDEX (not value) for O(log n) determination; even if
  monotonicity breaks locally, the bisection result stays conservative
  (never over-accepts) — safe default for w-node-style prefix acceptance.
- Schema/artifact evolution: make new certification-critical fields
  REQUIRED positional (no default) so pre-migration artifacts hard-refuse
  (KeyError/False) instead of silently certifying with missing data;
  enforce new validity caps at the single internal accessor chokepoint so
  every external accessor inherits the guard for free without a new
  sentinel.
