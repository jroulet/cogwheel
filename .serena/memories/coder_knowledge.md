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
- NEVER hand-tune a calibration/curvature constant to make your own
  certificate pass (self-grading). When a plan pins a formula's STRUCTURE
  but not an O(1) constant, hard-code the unknown at ONE named edit site
  with a loud placeholder note and flag it as owed expert work. Corollary
  (Build 8h-c1): a guard whose threshold is DERIVED from that same unpinned
  constant is self-referential — it passes exactly where the constant is
  wrong, so it cannot be cited as evidence the rung is accurate.
- UNSHIP, DON'T PATCH: when review shows a new rung is uncertified, remove
  it from the live serve path (slot returns None -> falls through to the
  certified exact engine), revert its census category, and leave the module
  dormant/unwired with a STATUS docstring naming the re-enable condition.
  Deriving the missing constant is expert work, not a Coder guess.
- After a regex-anchored method retirement, re-parse the file (ast.parse):
  substring anchors can leave a silent IndentationError.
- Cubic splines are C2: put a NODE on each C2 kink, never a segment break;
  interpolate only the single smooth object (demodulated envelope) and
  rebuild analytic/switched parts closed-form at dense samples.
- LOO adaptive refinement: held-out error from a few-nearest-OTHER-node
  fit, normalized in the gate's error currency, hard-coded threshold.
  Keep ONE shared refinement loop parameterized by a node_error closure.
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
  set; the object rides pickle in __dict__.
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
  classification gate EXACTLY mirror the frozen path's gate. A SINGLE
  fast-path intercept at the top of the expensive method that returns None
  on every guard miss lets the exact path fall through untouched. When
  generalizing a parity/domain-restricted closed form to a wider domain,
  wrap the restricted term in abs()/sign-canonicalize and prove old-domain
  byte-identity via the ALGEBRAIC identity (abs() is identity when the
  original argument was already positive) rather than a rerun; only branch
  explicitly where the underlying physics changes sign (e.g. a Morse phase).
  Same technique extends a validated cascade one order further: factor the
  shared lower-order computation into a private helper returning all
  intermediate terms, keep the existing public assembly line-for-line
  identical, and prove byte-identity via the same side-by-side HEAD sweep
  before adding the new public function on top (Build 1c y''' cascade).
- SINGLE-SOURCE A CONVENTION: when you find an inline re-expression of a
  rule a primitive already owns (e.g. `np.sort(d - d.min())` beside
  `_frame_delays`), route it through the primitive. Byte-identity is proved
  by the ALGEBRAIC identity (float() of a np.float64 min is the exact same
  64-bit value), not by a rerun. Forward references to a helper defined
  later in the module resolve at call time — fine.
- A shared derivation helper dedupes CODE but not RUNTIME: if several call
  sites each invoke it, an expensive underlying computation (e.g. a full
  geometry sweep / image quartic solve) still runs once PER CALLER. For
  values derived from a fixed input pair, compute ONCE and carry them as
  fields on the partition/dataclass instead of re-deriving them inside
  hot-path functions.
- TRAIN/SERVE CONSISTENCY: when a guard is monotone over the band, gate at
  the band's WORST case and use the SAME worst-case point in every consumer
  (serve, census, training) — gating census at the best case over-attributes
  configs to a rung that serve actually refuses. Deviating from WP text for
  this reason is correct, but flag it loudly.
- Tighten a tolerance ONLY in a sub-region without touching the certified
  hot path: gate the constant on a PURE fn of the candidate params at the
  single shared decision site; the unchanged branch returns the OLD constant
  verbatim so cache purity holds. Key on the PHYSICALLY correct variable.
- Independent oracles for singular integrands must be regularized: a naive
  Int_0^inf t^{s-1} h(t) form is ill-posed — use subtract-h(0) or IBP-with-
  h'. A DIFFERENT regularization scheme from the code's is the point (F002
  non-circular); phase agreement also confirms sign/conjugation convention.
- Serena `replace_symbol_body` on a function target: the new body MUST
  include the `def` signature line — omitting it deletes the def+docstring,
  producing a column-0 IndentationError. When a literal replace_content
  needle is ambiguous across sibling classes, prefer replace_symbol_body
  (unambiguous by name path) over widening the needle.
- For large Monte-Carlo census sweeps (N>=1e5), stream results into fixed
  threshold-grid histograms (`counts_ge += (arg>=grid).sum()`) instead of
  storing per-sample arrays — the histogram IS the CDF, no memory blowup.
- To classify which guard blocks a sample, toggle ONE guard off via
  `dataclasses.replace` on a frozen config object and re-call the real
  guard function — never re-derive the guard math inline.
- SDK caps inlined short-term memories at 24KB (tail-kept); earlier entries
  survive only in git history, not the prompt.
- Prefer OPTIONAL trailing args with backward-compatible defaults over
  changing a function's return-tuple shape when adding capability.
- When a boundary/refusal is monotone over an ordered grid, bisect on the
  node INDEX (not value); even if monotonicity breaks locally the result
  stays conservative (never over-accepts).
- Schema/artifact evolution: make new certification-critical fields
  REQUIRED positional (no default) so pre-migration artifacts hard-refuse
  instead of silently certifying; enforce new validity caps at the single
  internal accessor chokepoint.
- A byte-identical default/fallback branch must keep the LITERAL original
  expression, not a call into the new generalized helper fed a degenerate
  input — the generalized form can be FP-close but not bit-identical.
- For non-circular/star-shaped admission regions (e.g. sheared lobes),
  normalize by a DIRECTIONAL per-angle boundary function, not a scalar
  reach/radius constant — near-cusp clearance is far below the far-cusp
  scalar extent.
- Tell Test Dev when a refusal is bounded BELOW by an intrinsic quantity
  (e.g. complex-norm separation >= |Im(x_c)|): mutating one input cannot
  force the refusal branch, so reachable-red requires the genuine physical
  regime, not a monkeypatched value.
- A partially-rotated frame is a silent correctness trap: if serve
  de-rotates some coordinates (y1,y2 into the eigenframe) but passes another
  angle un-rotated, an off-axis parameter yields finite-but-wrong output.
  Guard every parameter axis the emulator/approximation was trained at by
  mirroring the existing axis guard exactly (`if lens['x'] != 0: return None`).
  Same discipline applies to a multi-step geometry pipeline that isn't
  frame-rotated but IS branch/parity-tagged: thread the SAME branch/parity
  value through every call (seed frame -> root-find frame -> reconstruct);
  don't let an intermediate call re-derive or default it (Build 1c).
- When a node-dependent carrier phase (e.g. exp(±iw t_min)) must be applied
  symmetrically on both the producer and reconstruct sides, factor it into
  ONE shared `_frame_phase(w, t_min)` helper both call — prevents a mod-2pi
  or sign asymmetry between the two sides (Build 8h-d2).
  Compose a multi-stage serve gate cheapest/most-discriminating-first
  (region box -> band -> corridor/exclusivity -> fine containment ->
  exclusion balls -> count/floor checks), and make an eigenframe/optional
  arg's "not supplied" case decline via an explicit `isfinite` precondition
  rather than relying on bare NaN-comparison-is-False semantics.
- To tell a genuine discontinuity/kink from grid aliasing, refine node
  density (e.g. 4->6->8->12) and confirm the jump does NOT shrink; also
  check whether narrowing the domain range removes the trip. If neither
  changes it, it's a real feature of the function, not an artifact.
- When a mandated fix ("route through the shared inverter") is algebraically
  equivalent to what a tight legacy tolerance was already measuring, and
  replaying it as a round-trip reintroduces FP cancellation on a near-
  degenerate fixture, the two mandates ("use the shared path" and "don't
  weaken the tolerance") are mutually incompatible for that fixture —
  escalate the conflict rather than silently picking one side.
- Exact literal complex constants for known phase/parity factors (e.g. a
  Morse index i^n) — hardcode the literal (-1j, 1.0) rather than evaluate
  via cmath.exp/trig; sub-eps round-off (~1e-16) in the transcendental form
  can break a downstream flat-magnitude pin invariant.
- Two mutually-dependent modules (parent imports child at module load;
  child needs a parent-only helper) — break the cycle with a function-local
  (lazy) import inside the child function, not a restructure.
- When a fall-through/enum-like category tuple gains a member, fix any test
  that hardcoded per-category sample counts to derive them from
  `len(category_tuple)` dynamically instead of a magic number, so future
  category additions don't silently rot the count.
- When a plan/brief hands you a closed-form derivative formula, re-derive
  it by hand rather than transcribing verbatim, then verify against an
  independent multi-precision numeric oracle (e.g. mpmath) before shipping
  — a brief's formula can carry a sign/factor typo (F038: brief said
  p=-u, correct is p=(lam±gamma)-lam*u) that only hand-derivation catches.
- When an old finite-epsilon numerical probe and a new exact closed form
  disagree at a boundary config, use a higher-resolution/exact direct
  computation (e.g. brute-force image-finding at very small epsilon) as
  the tie-breaker oracle rather than assuming the old probe was right —
  it may have mislabeled the boundary itself (Build 1b fold-orientation).
- Replacing a finite-diff derivative with an analytic closed form already
  available on a sibling primitive (e.g. `geometry.caustic_derivatives`
  for a tangent that used to be a 1e-6 forward diff): confirm the new and
  old vectors point the SAME direction (dot/cross sign) before shipping —
  magnitude/perpendicularity agreement alone can hide a silent orientation
  flip that only a downstream sign-dependent consumer (inward_sign) exposes
  (Build 1d).
- A brief giving reach and direction as SEPARATE formulas (with e.g. an
  |eigenframe_point|^2 ≠ reach^2 because of a positive factor) is correct
  by design — the normalized direction is unaffected by the factor. Don't
  unify them into a single formula unless the brief explicitly endorses it.
- When a probe fan must be reflection-invariant (e.g. min-over-angles
  w_cert where the underlying geometry has a reflection symmetry), make the
  angle set symmetric: tuple(k*pi/8 for k in range(-4,5)) rather than a
  one-sided [0..pi/2] sweep — proof: symmetric set is invariant under R, so
  min is R-invariant and direction-canonicalization no longer matters.
- Identity default for an arc-length map is NOT bit-identical to HEAD
  raw-theta spline (matches ~5e-15 due to B-spline translation invariance);
  fine for tolerance-based suites but a bit-exact-vs-stored-HEAD assertion
  would drift — document this seam, don't assert bit-identity vs HEAD.
- When a new required field changes the semantics of other stored fields
  (e.g. knots now in s vs raw theta after an arc-length reparametrization),
  an identity-map fallback on load is WRONG — old artifacts must hard-refuse
  (KeyError) and be retrained. Do NOT add a fallback that silently serves
  at the wrong coordinate offset.
- Cell midpoints of tiny synthetic test charts are not valid interpolation
  certificates for production accuracy bars — only node-exact round-trips
  and physical off-grid witnesses with matching chart coverage are valid
  certificates.
