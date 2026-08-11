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
- DD-PRODUCT CEILING PATTERN (Build wedge_followup, 56a223a): cap `w_range[1]`
  in `from_wedge_engine` via `dd_w_cap = _DD_PRODUCT_MARGIN / (r_grid[-1] *
  reach_max)` where `reach_max = max(r_table[:, theta_mask])` over gamma_grid ×
  theta nodes within the tile's theta_wedge_range. Use `r_grid[-1]` (r_max),
  NOT r_min — r_max gives the tightest (most conservative) global bound ensuring
  `w_max * r_max * reach_max <= 58` at the worst-case corner node. The brief may
  say r_min; this is a brief error — the correct deviation is to use r_max.
  Move `_log_w_grid()` AFTER this cap so the grid is built from the already-
  capped w_range. If cap < w_min, `_log_w_grid` raises ValueError cleanly.
  Mirror the constant: surrogate.py cannot import from surrogate_training.py,
  so define a LOCAL `_DD_PRODUCT_MARGIN = 58.0` with a comment noting it mirrors
  the training module's constant.
- ARC-LENGTH MAP PATTERN (Build wedge_followup, 56a223a): in `from_wedge_engine`,
  compute theta→s map via: `rep_gamma = np.median(gamma_grid)`, `arc_theta_fine =
  np.linspace(theta_wedge_range[0], theta_wedge_range[1], 2001)`, `arc_speed =
  geometry.caustic_speed(rep_gamma, arc_theta_fine, branch=1)`, `arc_s_fine =
  cumulative_trapezoid(arc_speed, arc_theta_fine, initial=0.0)`, `theta_to_s =
  np.vstack([arc_theta_fine, arc_s_fine])`, then `s_grid = np.interp(
  theta_wedge_grid, arc_theta_fine, arc_s_fine)`. Key invariant: `arc_theta_fine`
  and `theta_wedge_grid` share EXACT endpoints (both from `theta_wedge_range`),
  so `np.interp` never extrapolates. Pass `theta_to_s=theta_to_s, s_grid=s_grid`
  to `from_wedge_values`. The 2001-node fine grid matches `_FARFIELD_ARC_MAP_SIZE`;
  avoid introducing a separate local constant that silently duplicates it
  (INS-w3-001: local `_ARC_MAP_NODES = 2001` harmless but should reference the
  module constant). SUPERSEDED for the wedge chart by the cusp-adapted axis
  pattern below (2026-08-07 WP3) — Tube/Lobe/FarField still use this genuine
  arc-length pattern unchanged.
- SCHWINGER CEILING IS INDEPENDENT OF DD CAP: the DD cap prevents training nodes
  from exceeding `w * |y| = 58` (the diffraction-delay product above which double-
  double quadrature cannot maintain 1e-10 accuracy). However, the engine has its
  own Schwinger certification ceiling (~w~60 at large |y|) that can refuse nodes
  even below the DD cap. Success rate depends on both; the DD cap's job is solely
  to prevent IMPOSSIBLE requests, not to guarantee all nodes pass the engine's
  independent Schwinger gate. Do NOT interpret a low success rate (e.g. 6%) as a
  failure of the DD cap formula — verify the FORMULA (DD product invariant) rather
  than the success rate.
- ARC-LENGTH REMAP SERVE-TIME CONSISTENCY: both training and serving use the SAME
  monotone `theta_to_s` table, so the spline's grid-node exactness property is
  preserved through the remap (training stores `s_grid = interp(theta_wedge_grid,
  theta_fine, s_fine)`, serving computes `s = interp(theta_query, theta_fine,
  s_fine)` using the saved table). Node-exact accuracy budget: ~6e-9 interp error
  at 2001 nodes, so widen `_NODE_EXACT_TOL` from 1e-10 to 1e-7 and document the
  budget arithmetic at the constant's definition.
- INTERIOR WEDGE CHART IMPLEMENTATION (Build interior_wedge_chart): pattern
  for adding a new chart type to `cogwheel/lensing/surrogate.py`: (1) Add
  axis schema constant + known-schemas set. (2) Add map dataclass (frozen,
  holds precomputed tables) + validator. (3) Add _to_fixed / _from_fixed
  coordinate transforms (must be exact inverses — verify round-trip to
  ~1e-16). (4) Add chart dataclass (frozen) with `from_*_values` classmethod
  (fits tensor spline) + `_assemble` classmethod (NPZ load, no re-fit).
  (5) Add `_*_serves` gate function (cheapest-first gate ordering). (6) Add
  loop in `select_chart` at appropriate priority. (7) Add branch in
  `_evaluate_chart`. (8) Update `LensAmplificationSurrogate.__init__`
  isinstance tuple (EASY TO FORGET — causes ValueError on construction).
  (9) Add `_chart_to_npz` / `_chart_from_npz` branches. (10) Add
  `from_*_engine` training entry point. (11) Add provenance builder.
  The D₂ fold (abs(y1), abs(y2)) is the correct quotient for the astroid
  caustic symmetry; theta_wedge = atan2(|y2|, |y1|) in [0, pi/2].
- FOLD-CORRECTED ppGO (Build fold_corrected_ppgo): `fold_ppgo_correction`
  in `chang_refsdal/_airy_fold.py` replaces raw `geometric_amplification`
  for degenerate-delay pairs near folds. Design: structural-gates-only (no
  error-estimate or ETA_MAX gate) — falls back to raw ppGO on any gate miss.
  Uses LAZY import of `geometric_amplification` from `operator` to break the
  circular import (_airy_fold is imported by operator at module level).
  Wiring: (1) `_measure_cell` in ppgo_map.py uses fold_ppgo_correction
  (deferred import inside the evaluate closure). (2) `born_carrier_from_
  partition` in channels.py adds an additive fold correction in the
  above-split non-saddle else-branch, AFTER ppgo_plus_ghost assignment.
  On any structural refusal the correction is silently skipped (carrier
  stays byte-identical to uncorrected). Non-finite airy values guarded with
  np.where(finite_mask, ...).
- ppGO INTERIOR EXTRAPOLATION (Build ppgo_interior_certification):
  `_extrapolate_floor` in ppgo_map.py uses power-law error decay
  (log-log linear fit) to extrapolate the w_floor for interior cells
  (rho_center < 1.0). Constants: _EXTRAP_MIN_NODES, _EXTRAP_R2_THRESHOLD
  (0.9), _EXTRAP_MAX_RATIO (5.0), _EXTRAP_W_CERT_DEFLATION, _EXTRAP_N_FIT.
  Self-falsification: refuses when R-squared < threshold (beat aliasing) or
  extrapolated ratio > MAX_RATIO (wild extrapolation). Interior cells get
  a relaxed `floor > w_ceiling` guard (extrapolation can push floor beyond
  the DD product ceiling). `interior_w_nodes_per_decade` field on
  TrainingConfig (default 15) wires through _train_band_charts via a 3-way
  if/elif/else (tile override -> interior -> else config.w_nodes_per_decade).
  Fixed _subdivide_farfield_tile to use same 3-way pattern.
- When a new required field changes the semantics of other stored fields
  (e.g. knots now in s vs raw theta after an arc-length reparametrization),
  an identity-map fallback on load is WRONG — old artifacts must hard-refuse
  (KeyError) and be retrained. Do NOT add a fallback that silently serves
  at the wrong coordinate offset.
- Cell midpoints of tiny synthetic test charts are not valid interpolation
  certificates for production accuracy bars — only node-exact round-trips
  and physical off-grid witnesses with matching chart coverage are valid
  certificates.
- SIMPLENS ADAPTER PATTERN: when a serve path needs to call a helper that
  expects a rich partition-like namespace (with source/gamma/matrix etc.)
  but the available object (e.g. a geom struct) lacks those fields, wrap it
  in a `types.SimpleNamespace` with the required fields explicitly set —
  do NOT add those fields to the geom object itself. Document which fields
  the helper reads so the adapter is an exhaustive contract, not a guess.
- GEOMETRY DATA TYPE GOTCHA: `geom.images` is a LIST not ndarray (despite
  the type annotation suggesting otherwise) — use `list(geom.images)` not
  shape-based indexing. The `macro_matrix` is NOT stored on geom — reconstruct
  it via the `macro_matrix(...)` function from the partition.
- LOW-W FLAT EXTRAPOLATION PATTERN (Build low_w_extrapolation): to serve
  draws with w < chart.w_min via flat extrapolation, (1) add a
  `_log_w_band_serveable` gate that checks ONLY the high end
  (`log_w_max <= chart.log_w_grid[-1]`), replacing the old bilateral
  `_log_w_band_inside` at all 5 call sites; (2) clamp with `np.clip` in
  `_evaluate_chart` BEFORE the B-spline call (`log_w_query = np.clip(
  log_w_query, chart.log_w_grid[0], chart.log_w_grid[-1])`). Use
  `np.clip` (not `np.maximum`) for the one-sided low-end clamp so the
  high-end guard stays strict. The physics justification: KERNEL_SUM
  envelope → 0 as w → 0; DIFFRACTIVE/INTERIOR → sqrt(mu_macro); flat
  clamp is O(w_min²) error. scipy BSpline extrapolates polynomially
  off-grid — the clamp MUST precede the spline call, not follow it.
  Accuracy bar: w_min/2 within 3e-3 of max|F|.
- BORN RESIDUAL SPARSE-GRID TRAINING PATTERN: the Born residual chart is
  trained on a sparse grid (not the full Born grid) because the residual
  (exact F − Born) is smooth and well-resolved by fewer nodes. Training
  entry point follows the same `from_*_engine` / `from_*_values` pattern
  as other charts. The sparse grid must still satisfy >= 4 nodes per axis
  (axis node count constraint). Census classifies residual-served draws
  with `served=True` and a residual-specific `serve_method` indicator.
- CUSP ARM BINARY-SEARCH BOUNDARY MEASUREMENT PATTERN (Build
  cusp_arm_boundary): measure the actual serve/refuse boundary of
  `cusp_amplification` by calling it directly (not an R-gate formula).
  Use `max(all_refused)` not `last-consecutive` to find the conservative
  boundary (monotonicity mostly holds but may break locally). Floor result
  to 2 decimal places (conservative direction) and set `_CUSP_ARM_COVERAGE`
  in surrogate.py. Script outputs delta_theta in radians (same units as
  `_tube_serves` consumption). Refinement pass: re-run at worst-case config
  with N=10000. Exclude saddle parity (converges to 0 due to deep-interior
  images). After enabling `_CUSP_ARM_COVERAGE=0.07`, existing cusp_window
  tests are unaffected because residual = max(0, window - 0.07) still
  blocks queries at exact cusp vertex (delta=0).
- MPMATH LAZY IMPORT + PAIRED N/2N CERTIFICATION PATTERN (Build
  schwinger_qd): for an mpmath quadrature extension above a DD ceiling:
  (1) lazy-import mpmath inside the function body (`import mpmath as mp`
  at first call) to keep it an optional dependency; list under
  `[project.optional-dependencies] training` in pyproject.toml. (2) Set
  `dps = 30 + ceil(w)` for the working precision. (3) Certify on
  RECONSTRUCTED F (not the raw integral value) via paired N/2N
  evaluation: `|F(2N) - F(N)| / |F(N)| < tol`. (4) Dispatch in the
  public function: check QD ceiling (150) first, then DD ceiling (60);
  fallthrough to double-double. (5) W_CEILING_SCHWINGER_QD=150.0 exported.
  KEY BUG: `mp.linspace` must receive `mpf` endpoints — passing `float()`
  casts causes catastrophic precision loss and ~1e4 magnitude errors.
  Training pipeline: `_SADDLE_W_CEILING` raised to 148.0 (2 below QD
  ceiling as safety margin); routing pivot in `_saddle_grid` /
  `_positive_parity_grid` changed from DD ceiling (60) to QD ceiling (150).
  Sequential batch (not parallel) for the mpmath band: `f_schwinger`
  dispatches internally so the call site is unchanged.
- MIN_GAMMA_BAND = 1e-6 (NOT 0.0): setting `min_gamma_band = 0.0` in
  `stable_gamma_bands` triggers near-infinite bisection at degenerate
  boundaries — the function terminates via float-resolution bisection (1066+
  iterations for a width-0.004 sliver at gamma=0 or saddle near-gamma=1
  topology transitions). Use `min_gamma_band = 1e-6` as the production
  value; this closes region 10 (phantom slivers) without the bisection
  blowup. F041 test constant `_F041_MIN_WIDTH = 0.02` intentionally stays
  (tests the arc-guard fix, not this threshold). The 3 edit sites are:
  `TrainingConfig.min_gamma_band` default, `stable_gamma_bands` default
  arg, and `scripts/measure_dropped_slivers.py` MIN_WIDTH constant.
- CENSUS DRY-RUN METHODOLOGY (census_dry_run): a geometry-only coverage
  audit without trained charts. Sample N=10K draws from the full parameter
  space (gamma, |y|, θ, w log-uniform). Use `geometry_partition` (cheap
  quartic, no engine) with minimal w_grid (2 pts). Classify each draw by
  structural gate only: born (rho>1), tube_feasible (eta in
  f_floor*Rc..f_max*Rc), wedge_feasible (rho<1, w*|y|<=58),
  ppgo_fold (rho<1, 4 images, ξ>=4), cusp_arm (within cusp window above
  coverage threshold), exact_engine (residual). Stream counts into
  threshold-grid histograms (not per-sample arrays). 100% structural
  coverage means every draw reaches at least one gate — confirms no
  uncovered region before launching production training.
- GENERIC TILE SUBDIVIDER `_subdivide_tile` (Build
  subdivision_recursion_and_coordinate_cleanup, 2026-08-07): unifies
  `_subdivide_farfield_tile`/`_subdivide_wedge_tile` behind one helper
  parameterized by (child-box splitter, build_child, admit_child); bounded
  recursion via a module constant `MAX_SUBDIVISION_DEPTH` (=3), recursing
  full subtrees and accumulating `total_packed` from each subtree. Each
  child summary entry gains an ADDITIVE `achieved_depth` key; the return
  dict adds `max_achieved_depth` on top of the pre-existing keys — never
  removes/renames existing keys. `CarrierDiscontinuityError` is caught and
  treated as a terminal ladder gap, never recursed. The two original
  functions become thin wrappers building closures over the module's own
  `_build_*_chart` + held-out probe helpers, signatures and all call sites
  (`_train_band_charts`) unchanged.
- WEDGE FIELD RENAME IMPLEMENTATION (WP3, same build):
  theta_to_s->theta_to_u, s_grid->u_grid on `InteriorWedgeChart` ONLY
  (Tube/Lobe/FarField charts keep genuine arc-length theta_to_s/s_grid
  untouched — DO NOT rename those). DRY validator core
  `_validate_axis_map(arr, grid, *, ordinate_name)` checks ONLY shape
  (2,N>=2) + finite + row0 strictly increasing from grid[0] + row1
  strictly increasing from ~0 — NO magnitude/length-scale bound (u=d**(2/3)
  has different magnitude than a true arc-length s over the same tile).
  `_validate_theta_to_s`/`_validate_theta_to_u` both delegate to it.
  Schema bump hard-refuses the OLD tag (no `_KNOWN_..._SCHEMAS` overlap);
  the renamed field becomes a REQUIRED NPZ key on load (no optional
  fallback) — any synthetic/legacy artifact built on the identity/None
  path now hard-refuses (KeyError) and must be rebuilt with a real map.
  Domain guard: raise (never clamp) outside the physical domain
  ([0, pi/2] here) — clamping would silently serve the reflected/adjacent
  tile's basin and mask a caller bug.
- CLOSED-FORM r_caustic (same build): replaced a 720-point dense scan with
  brentq bracket+refine on the parametric caustic curve r(u)=|y(u)|;
  bracket count keyed to topology (48 nodes astroid / 720 saddle, by
  parity). The `n_sample` parameter (and its `<16` validation) is fully
  retired — delete rather than deprecate-in-place when a plan explicitly
  says the parameter is dead. gamma=0 and parity-boundary/saddle-miss
  configs must raise the NAMED domain error (`LensDomainError`) directly
  from the root-finder — audit for any bare arithmetic (e.g. division by a
  vanishing gamma) upstream of the raise that could leak a raw
  `ZeroDivisionError` instead.
- EXTERIOR POLAR CHART IMPLEMENTATION (Build exterior_polar_rechart,
  4d59a6d + 0a31fcf): ExteriorPolarChart is a frozen dataclass with
  (rho_grid, theta_c_grid) axes and NO arc_map — the polar coordinate is
  single-valued so no arc-length map is needed (tile edges sit on cusp
  rays, `_exterior_polar_tiles`; no tile straddles a cusp). Gate
  `_exterior_polar_serves`: np.isfinite guard + box containment +
  exclusion balls + image_count + eta. `from_engine` takes polar
  (rho_range, theta_c_range), no interior path. `select_chart` priority:
  tube > exterior_polar > lobe > wedge. NPZ kind='exterior_polar' in
  `_chart_to_npz` / `_chart_from_npz`. WP2 rewire: deleted the
  `_farfield_box_to_smooth` bridge + `_saddle_arc_branch`; removed the
  parity!=1 refusal from `_build_farfield_chart` (BOTH parities chart:
  astroid + macro-saddle exterior via additive scalar-reach rho since the
  deltoid lobes don't enclose the origin); added
  `_CUSP_EXCLUSION_DISTANCE=0.2` y-units (sized by the separation-gate
  contour, WIDER than `_CUSP_ARM_COVERAGE=0.07` image-theta rad) +
  `_exclude_near_cusp`; `_load_or_build` catches schema mismatch. Saddle
  axis edges NOT aligned (deltoid off-axis). `_KNOWN_ENVELOPE_DEFINITIONS`
  widened so the loader accepts the union of far-field AND interior
  envelope tags.
- EXTERIOR POLAR CUSP-ADAPTED U COORDINATE IMPLEMENTATION (Build
  exterior_polar_cusp_coordinate, 1a97bbd, 2026-08-08): added optional
  `theta_to_u: np.ndarray | None` field to ExteriorPolarChart, mirroring
  the lobe/wedge pattern: from_values/from_engine/_assemble all accept
  optional theta_to_u/u_grid; when provided the spline is fit on a UNIFORM
  u_grid; `_evaluate_chart` remaps theta_c->u via np.interp before spline
  contraction; `_chart_to_npz` writes conditionally (only when not None),
  `_chart_from_npz` reads via `data.get(...)` (returns None on a missing
  key, matching the field's own type annotation — the OPPOSITE of the
  wedge/lobe REQUIRED-key convention, where the field is read
  unconditionally and absence hard-refuses). Axis schema bumped
  'exterior_polar_rho_theta_c' -> 'exterior_polar_rho_u_v1'. BOTH
  `_build_farfield_chart` (parity==1 tiles) and `_subdivide_tile`
  children pass the cusp-adapted map via `_wedge_cusp_axis_map`; the
  macro-saddle exterior (parity==-1) passes None (raw-theta fallback).
  Added `_uniform_axis` to surrogate_training.py imports. FIX INS-1-001:
  `data.get(prefix+'theta_to_u')` in ALL THREE `_chart_from_npz` branches
  (lobe, wedge, exterior-polar) so theta_to_u=None charts survive an NPZ
  round-trip. FIX INS-2-001: schema test renamed to
  test_new_schema_without_theta_to_u_loads_with_none (asserts None-load,
  not KeyError). NOTE (INS-4-001, inspector): the WEDGE loader must NOT
  use `.get()` — wedge v3 requires theta_to_u (from_wedge_engine always
  builds it, `_chart_to_npz` always writes it); the hard KeyError on a
  corrupt artifact is the required contract. Change wedge line back to
  `data[prefix + 'theta_to_u']` (exterior-polar + lobe keep `.get()`).
- REGIONS FILTER (same day, remeasure_v3 WP1): `_train_band_charts` gains
  `regions: tuple[str, ...] | None = None` (None → full tuple ('tube',
  'exterior', 'wedge_interior', 'lobe_interior')); each section is
  region-guarded with else-defaults (exterior_tiles=None,
  region_exclusion_rho=exclusion_rho); `exterior_admission=None` is
  initialized BEFORE the exterior guard to avoid a NameError in the
  unguarded dispatch loop (iterates `admitted`, empty for skipped
  regions). `train()` threads the kwarg; `scripts/train_lens_surrogate.py`
  gains `--regions` nargs='*'. Combined with `m_lens_range` this makes a
  per-region probe a REAL single-stratum production-path call.
- PROBE ARTIFACT READING (driver probes, post-strand): chart.provenance
  in-memory LACKS heldout_eps after an NPZ load — read heldout_eps from
  the NPZ provenance (on-disk artifact), never the in-memory object; an
  all-NaN held-out-eps reading can be a probe bug (wrong config scale),
  not a coordinate failure. Probe config must match the production
  tiling it claims to re-measure (gamma_band_halfwidth 0.04, NOT 0.48);
  a wide band invalidates the comparison.
- LOBE SUBDIVIDER IMPLEMENTATION (Build saddle_forensics, 2026-08-08):
  `_lobe_child_boxes` + `_subdivide_lobe_tile` — a THIRD wrapper over the
  shared `_subdivide_tile` (after far-field + wedge), threading a lobe key
  into the child tile dict; wired into `_train_band_charts` lobe branch so
  subdivision replaces the immediate ladder_served_gap. INS-1-001: build_child
  must pass `w_nodes_per_decade=eff_w_nodes` (3-way resolve: tile override ->
  config.interior_w_nodes_per_decade -> config.w_nodes_per_decade) NOT the raw
  `w_nodes`, matching the wedge subdivider pattern — lobe-interior children
  get the interior node density. `_LOBE_CUSP_EXCLUSION_DISTANCE=0.1` was added but then RETIRED
  (deleted, 98c4e7f) by the follow-on cusp-adapted-coordinate build: the
  existing eta_max tube-shell nearest-distance test in
  `_SaddleLobeAdmission.admits` already rejects near-cusp-vertex tiles (cusp
  vertices are in caustic_cloud), so the carve-out constant was never
  functionally needed — no explicit carve-out code shipped.
- PPGO ABOVE-CEILING ENGINE-INTERCEPT RUNG (Build ppgo_above_ceiling,
  2026-08-08): `_ppgo_above_ceiling` method + intercept in
  `_amplification_coefficients` BEFORE the engine eval. Imports
  W_CEILING_SCHWINGER_QD (150.0) from `_schwinger` and RHO_END (4.0) from
  operator. Gate: w_max > 150 AND min_delta_tau > 0 AND
  w_lo*min_delta_tau >= 4.0. Whole-band serve via fold_ppgo_correction +
  reconstruct_farfield(FARFIELD_KERNEL_SUM). LensDomainError propagates
  unswallowed; non-finite guarded; fallthrough -> SchwingerCertificationError
  unchanged.
- LOBE CUSP-ADAPTED COORDINATE IMPLEMENTATION (Build lobe_cusp_coordinate,
  98c4e7f, 2026-08-08): `_lobe_cusp_axis_map` (uniform-in-u `(2, 2001)`
  `[theta_fine, u_fine]`, same node count as `_FARFIELD_ARC_MAP_SIZE`)
  mirrors `_wedge_cusp_axis_map`; renamed theta_to_s -> theta_to_u /
  s_grid -> u_grid on `LobeInteriorChart` ONLY (field, docstring,
  from_lobe_values params, _assemble, _evaluate_chart lobe branch,
  _chart_to_npz / _chart_from_npz lobe branches, _build_lobe_provenance);
  single schema tag `_LOBE_AXIS_SCHEMA_NEW='lobe_caustic_relative_v1'` in
  `_KNOWN_LOBE_AXIS_SCHEMAS` — old V1 and sqrt-edge tags hard-refuse at load,
  no identity fallback; `_LOBE_ARC_MAP_SIZE` deleted (unused). INS-3-001
  fix: cusp-adapted u-coordinate was BUILT but NEVER ACTIVATED in production
  (build_lobe closure + _subdivide_lobe_tile.build_child called
  `_build_lobe_chart` WITHOUT cusp_angle/cusp_side -> raw-theta fallback
  won); fix = derive nearest-cusp+side via the SHARED `_lobe_nearest_cusp`
  helper (single source with `_lobe_child_boxes`) at BOTH build sites.
  `_lobe_child_boxes` splits at the U-MIDPOINT (u=d**(2/3)) mapped back to
  theta_local via np.interp — NOT the raw theta midpoint. from_lobe_engine:
  cusp_angle=None keeps a raw-theta uniform grid fallback (backward compat);
  all tiles carry cusp angles in the production pipeline. Professor's
  non-blocking note: `_chart_from_npz` unconditionally accesses
  `data['theta_to_u']` (KeyError if absent) while `_chart_to_npz` only
  writes it when not None — a theta_to_u=None chart builds but cannot
  survive an NPZ round-trip (latent trap for external callers; not hit by
  the current pipeline).
- SIBLING-INSERTION CLOBBER (INS-4-003, lobe_cusp_coordinate build):
  inserting `_lobe_cusp_axis_map` adjacent to `_wedge_cusp_axis_map`
  accidentally DELETED the neighbor's `return theta_fine, u_fine` (caught
  by the Inspector at line 702, reinserted). After inserting a new
  function next to an existing one, verify the neighbor's body is
  byte-unchanged — same failure family as the replace_symbol_body
  signature-line gotcha.
- CUSP PPGO FAST RUNG IMPLEMENTATION (Build cusp_ppgo_high_w, 2026-08-08):
  the ppGO fast rung lives in `_pearcey_cusp.cusp_amplification`, positioned
  AFTER `radius = math.hypot(x, y)`, calling `fold_ppgo_correction` (via a
  lazy `_airy_fold` import) with a LensDomainError guard -> falls through to
  the Pearcey path. `_R_PPGO_ERROR_CONST=50.0` coordinated with
  `_UNIFORM_ERROR_CONST` (moves r_ppgo_min 54.3 -> 464.16). INS-5/INS-6
  fixes: raised `_PPGO_SERVE_W` to 20000.0, saddle-parity test fixtures to
  w in [5000, 20000], corrected stale comments. OWED: post-build driver
  measurement to tighten `_R_PPGO_ERROR_CONST`.
- EXTERIOR 2D (rho, u) FOLD-CARRIER (Build exterior_2d_fold_carrier,
  2026-08-10): ExteriorPolarChart field is `rho_u_carrier` (2D
  (n_rho, n_theta_c), default None) — Re(tau_c(rho,u)) at EVERY node, not a
  per-rho median. `_compute_rho_u_carrier` probes ghost_kernel per node,
  takes median over gamma, NaN-fills along u (axis=1) THEN rho (axis=0)
  (linear interp, zero-order hold at boundaries), all-NaN -> None.
  `from_values` demodulates exp(-1j*w*rho_u_carrier[None,None,:,:]) BEFORE
  the residual carrier_rate demod; serve re-modulates in reverse order with
  the delay BILINEARLY interpolated at the query u-coordinate (searchsorted
  on u_axis built from theta_c_grid via theta_to_u — NEVER raw theta_c).
  Schema V5 'exterior_polar_rho_u_carrier_v2' writes; V4
  'exterior_polar_rho_log_carrier_v1' stays known; backward compat: NPZ
  load tries 'rho_u_carrier' key, falls back to 'rho_carrier' broadcast
  1D->2D (np.broadcast_to) — broadcast at the LOAD boundary, single serve
  path. Index-pairing on theta_c_grid (u_grid[j] is partner of
  theta_c_grid[j]) needs no inverse interp. surrogate_training.py zero
  changes in this build.
- FOLD-CARRIER DEMODULATE RE ONLY: demodulate/remodulate ONLY Re(tau_c);
  never touch Im(tau_c) — a full-complex e^{+w*Im} remodulation is
  explosive (~19x at w=30). from_engine's continuity gate AND k_chart
  estimation must both run on the 2D-demodulated envelope when
  fold_carrier=True (never the raw envelope).
- CLAMPED-log_w RE-MODULATION (INS-1-001, exterior_2d_fold_carrier): the
  fold-carrier re-modulation must use np.exp(log_w_clamped) — the SAME
  clamped log_w the carrier_rate re-modulation uses — NOT
  np.exp(log_w_query); a mismatch breaks phase cancellation on low-w
  extrapolation queries.

- SADDLE EXTERIOR CUSP-ADAPTED COORDINATE IMPLEMENTATION (Build
  saddle_exterior_full_treatment, 238d21e, 2026-08-10): `_deltoid_cusp_axis_map`
  in surrogate.py generalizes the cusp-adapted u=d**(2/3) map to the macro-
  saddle deltoid cusps — an INTERIOR-anchor generalization of the edge-anchored
  `_wedge_cusp_axis_map` (deltoid cusps sit inside the tile range, not on a
  wedge edge). Straddle check returns None (raw-theta fallback); boundary
  validation [0, pi/2] raises ValueError; np.clip in the left-of-cusp branch
  guards against FP artifacts at the hi endpoint; fine-grid endpoints pinned
  explicitly (theta_fine[0]=theta_lo, theta_fine[-1]=theta_hi). Wired into
  `_build_farfield_chart` parity==-1 branch (surrogate_training.py): probe
  deltoid cusp rays via `_deltoid_cusp_source_angles(gamma_mid,
  config.n_caustic_samples)` (same median-gamma approach as parity==1's waist),
  pick the nearest to the tile centre, build the map ONLY when the nearest cusp
  ray is at a tile boundary (nearest == theta_lo or == theta_hi) — an interior
  nearest cusp would straddle and return None; fall through to theta_to_u=None
  when no candidates / interior cusp. `_subdivide_farfield_tile` children
  inherit automatically via `_build_farfield_chart`. Fixed stale 'Positive-
  parity only' docstrings in `_exclude_ghost_dominated` + `_needs_fold_carrier`
  (now both parities). WP2 parity-gated cusp-window: `_SADDLE_CUSP_ARM_COVERAGE
  =0.0` placed right after `_CUSP_ARM_COVERAGE`; `_tube_serves` dispatches
  coverage = _SADDLE_CUSP_ARM_COVERAGE if chart.parity==-1 else
  _CUSP_ARM_COVERAGE — positive-parity path byte-identical to HEAD.
  scripts/measure_saddle_cusp_arm_coverage.py mirrors the positive-parity
  measurement methodology (measure_cusp_arm_actual_boundary.py) for post-build
  calibration; left untracked.
