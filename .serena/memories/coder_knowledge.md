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
  EXTENDED 2026-08-12 (lobe_exterior region wiring build): default tuple
  grows to a 5-tuple adding 'lobe_exterior'; see the dedicated 2026-08-12
  section below for the full edit list.
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
  the Pearcey path. `_R_PPGO_ERROR_CONST` coordinated with
  `_UNIFORM_ERROR_CONST` to set r_ppgo_min; shipped at 50.0, progressively
  tightened 2026-08-12 (3.0 -> 1.0, see the 2026-08-12 section below).
  INS-5/INS-6 fixes: raised `_PPGO_SERVE_W` to 20000.0, saddle-parity test
  fixtures to w in [5000, 20000], corrected stale comments.
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
- PPGO RESOLUTION GATE IMPLEMENTATION (Build operator_routing_one_home,
  2026-08-11): `_PPGO_RESOLUTION_GATE = 4.0` module constant placed after
  `_PPGO_BAR_DIVISOR` (documents that it mirrors operator.RHO_END — a
  circular-import barrier prevents a direct import). In the ppGO rung block
  (inside `if (radius >= r_ppgo_min and ...)`), BEFORE the existing try,
  compute `delta_min` from the already-available `images` list:
  `delays = sorted(geometry.delay(image, source, matrix) for image in
  images)`, `delta_min = min(b-a for a,b in zip(delays[:-1], delays[1:]))
  if len(delays) >= 2 else 0.0`. The existing try/except now wraps the
  additional guard `if (_airy_fold._merging_fold_pair(images, source,
  matrix) is not None or w * delta_min >= _PPGO_RESOLUTION_GATE):` — the
  fold_ppgo_correction call happens ONLY inside this new conditional
  (_merging_fold_pair inside try/except: morse_index can raise
  LensDomainError). On gate miss `result = None` so the rung falls through
  to the Pearcey uniform form. Fold-pair + resolved-saddle nodes are
  BYTE-IDENTICAL to pre-change behavior; only unresolved saddle nodes are
  newly refused.

## 2026-08-12 builds (lobe_exterior region wiring, deltoid exterior geometry, cusp gap)

- LOBE_EXTERIOR REGION WIRING IMPLEMENTATION (Build
  lobe_exterior_region_wiring, Option A): ONE Coder WP, 5 edits in
  surrogate_training.py: `_self_estimate` default region tuple grows to a
  5-tuple + per-region cost dict entry `'lobe_exterior': 1` (same cost as
  lobe_interior); `_train_band_charts` default region tuple grows to match;
  saddle/lobe admissions gate widened `{'exterior'}` -> `{'lobe_interior',
  'lobe_exterior'}`; packing-loop tile-block gate widened
  `'exterior'` -> `'lobe_exterior'`; the parity==1 exterior block is
  UNCHANGED (byte-identical). scripts/train_lens_surrogate.py --regions
  choices extended to include 'lobe_exterior'. `_tag_kind` decodes the
  emitted NPZ infix `_fflobeext_` -> 'lobe_exterior' checked FIRST, before
  `_ff_` — neither string is a substring of the other so there is no
  leak risk, but ORDER still matters if a future tag is added that IS a
  substring of another.
- ADMITS_EXTERIOR NAME COLLISION (2-arg vs 3-arg, cross-referenced from
  both the geometry_fix and region_wiring builds): `_SaddleLobeAdmission.
  admits_exterior(center, half)` (2-arg, lobe-exterior packing) is a
  DIFFERENT METHOD with a DIFFERENT SIGNATURE from `_InteriorAdmission.
  admits_exterior(center, half, source_magnitude_max)` (3-arg, positive-
  parity exterior packing) — same method NAME on two different admission
  classes, not a shared/overridden method. Never copy call-site arg counts
  across the two classes.
- DELTOID EXTERIOR GEOMETRY FIX IMPLEMENTATION (Build
  deltoid_exterior_geometry_fix, re-planned 2026-08-12 — supersedes an
  earlier abandoned corridor_half-exclusion draft, do not resurrect it):
  new `LobeExteriorChart` dataclass charts the deltoid exterior in
  LOBE-LOCAL `(rho_lobe, u=d**(2/3))` coordinates, reusing
  `_lobe_cusp_axis_map` (not a new map). `theta_to_u` is present from
  construction and is a SOFT NPZ read (`data.get(...)`, may be None),
  mirroring LobeInteriorChart's convention exactly — NOT the wedge
  loader's hard `data[...]` read (a build-time doc comment briefly
  mis-stated this as "wedge convention... read hard"; corrected by
  Foreman-Lite, INS-7-001). The inter-lobe corridor is served directly by
  the +y1 lobe's exterior chart — no separate corridor-half exclusion
  band. Cusp exclusion reuses the existing `caustic_cloud` nearest-
  distance test with `eta_max = f_max * R_c` (same currency as the
  tube-shell admission gate). Admission: image_count==2, KERNEL_SUM label
  only (no ghost term — that is the separate cusp-gap build). Oracle =
  Schwinger engine, matching every other far-field-family chart.
- LOBE_EXTERIOR MINUS_GHOST + LOWERED-PPGO-RADIUS IMPLEMENTATION (Build
  deltoid_exterior_cusp_gap, Option C hybrid, 2026-08-12, WP-1 + WP-2):
  WP-1 tightened `_R_PPGO_ERROR_CONST` in `_pearcey_cusp.py` from 3.0 to
  1.0 (r_ppgo_min ~71.1 -> ~34.2), opening the mid-w ppGO band for both
  parities; docstring notes a post-build calibration sweep is still owed;
  the ppGO rung block is otherwise byte-identical. WP-2 added
  `force_minus_ghost: bool = False` to `_build_farfield_chart` and
  `d_exclude: float = _CUSP_EXCLUSION_DISTANCE` to `_farfield_tiles`; in
  the saddle branch of `_train_band_charts`, `_farfield_tiles` is called
  with `d_exclude=0.0` (no blanket radius exclusion) and the packing loop
  instead checks `_exclude_near_cusp(d_exclude=_CUSP_EXCLUSION_DISTANCE)`
  per-tile to set `force_minus_ghost=True` on near-cusp tiles only —
  letting the ghost-gate + eps bar decide admission rather than a coarse
  radius cutoff. The flag threads through the `build_ff` closure,
  `_subdivide_farfield_tile`'s `build_child`, and the `_subdivide_tile`
  child-tile dict, so subdivided children inherit it. When True, uses the
  FARFIELD_KERNEL_SUM_MINUS_GHOST envelope definition instead of
  FARFIELD_KERNEL_SUM. Astroid (parity==1) path is byte-identical to HEAD
  throughout.
- TRIAGE INS-1-001 FIX (deltoid_exterior_cusp_gap, 2026-08-12): WP-2's
  `force_minus_ghost` training was crashing on a per-node `GhostDomainError`
  because `farfield_envelope_from_partition` (which implements MINUS_GHOST)
  was called OUTSIDE the node-level `except _REFUSAL_ERRORS` guard in
  `from_engine` — only the tile-level far-field except clause caught
  `CarrierDiscontinuityError`. Fix, two parts, both required: (1) PRIMARY —
  move the ghost-gate call inside the existing per-node try/except so a
  `GhostDomainError` there becomes a refused POINT (preserves partial-tile
  coverage — the actual design goal); (2) SAFETY NET — also add
  `GhostDomainError` to the tile-level except tuple alongside
  `CarrierDiscontinuityError`, so a whole-tile miss routes to the
  ladder-served gap. CRITICAL: a ghost-gate refusal must NEVER be routed
  into `_subdivide_farfield_tile` — the refusal is a GEOMETRIC boundary
  condition (near-cusp), not a resolution problem; subdivision cannot cure
  it and would recurse pointlessly to the depth cap.
- SADDLE-ORIGIN-RHO-MISCLASSIFICATION 5-SITE GUARD FAMILY (Build WP-1,
  2026-08-12): saddle (gamma>1) sources with rho<1 near the origin were
  being misclassified by code that assumed rho<1 <=> astroid interior. Five
  parity-conditional guards added, ALL of the form "if parity is saddle AND
  [some rho/image-count condition], refuse/reclassify" — positive-parity
  paths are byte-identical at every site: SITE1 `likelihood.py
  _ppgo_cell_coords`: `parity=='saddle' and rho<1.0` -> return None. SITE2
  `likelihood.py` fold-ppGO interior handoff: `gamma>1.0 and
  image_count!=4` -> return None (skip handoff, fall through to exact
  engine). SITE3 `surrogate_census.py classify_fallthrough`: after the
  existing rho>1 "born" check, add `gamma>1.0 and image_count==2` ->
  'born' (saddle 2-image corridor sources are genuinely out-of-box, not a
  distinct category). SITE4 `surrogate_census.py characterize_sample`:
  after rho computation, saddle rho<1.0 -> rho=None (mirrors SITE1's
  refusal so census band-splitting can't see a bogus rho). SITE5
  `ppgo_map.py w_cert`: saddle rho<1.0 -> UNKNOWN (defense-in-depth; the
  shipped w_cert map DOES have certified values in this cell region, so
  the guard is load-bearing, not vacuous).
- SADDLE-INTERIOR-CUSP `_is_interior` DISCRIMINATOR FIX (Build WP-1,
  2026-08-12): replaced a broken origin-centered-polar `r_caustic`
  interior/exterior check (which fails near a deltoid cusp gap-angle where
  `r_caustic` itself can raise `LensDomainError`, or return the wrong
  caustic-boundary branch for a ring-shaped saddle interior) with an
  IMAGE-COUNT discriminator: `_is_interior = len(images) >= 4`. The
  saddle deltoid interior is topologically a RING (4 images), not a disk —
  a source can transition 4->2->4->2 images as gamma/theta sweep, so any
  origin-radius-based check is fundamentally wrong there; image count is
  the correct, cheap, always-available discriminator. DRY'd the sibling
  `interior_degenerate` check to reuse the same `_is_interior` flag instead
  of a separate inline `len(images) > 2`.
- REMOVED — DO NOT REINTRODUCE (2026-08-11, superseded within the same
  build): an "effective-x delay-gap fallback" was added to
  `cusp_amplification` to handle off-axis sources where
  `_real_stationary_points` yields fewer than 3 SPs (computing an effective
  x from the min Morse-0/Morse-1 delay gap). It was TOO BROAD — it also
  fired for conventional on-axis sources that correctly yield 3 SPs,
  corrupting them (17 test failures). Reverted whole-block to HEAD
  (71a7f73); the underlying off-axis interior-cusp gap this was meant to
  close is still open and needs a narrower, correctly-gated redesign, not
  a revival of this fallback.

## 2026-08-13 builds (ppgo_interior_certificate, fold_exterior_ghost)

- SERIES-COEFFICIENT PORT: when porting a reference derivation into the
  shipping layer, reuse the module's existing primitives (`hessian()`,
  `magnification()`) instead of transcribing the reference's own
  constructions, then validate the port against the ALREADY-SHIPPED closed
  form (`_series_coefficients` c1/c2 vs `saddle_coefficients`, agreed to
  ~1e-14 over 46 images). Two shipped derivations of the same quantity are
  a legitimate cross-check gate; a re-typed formula is not.
- A CERTIFICATE MUST SELF-REFUSE WHERE ITS SERIES DIES:
  `ppgo_error_estimate` returns None on `w_min <= 0` or any non-finite
  mu/c3, so near-critical images (|mu| -> inf as rho -> 1) refuse by
  construction instead of returning a small number. Measured cert ~5.7e6 at
  rho=0.99 — it explodes long before c3 goes optimistic, which is why the
  separate near-caustic leg could be dropped.
- REFUSAL GUARDS GO AT THE ENTRY POINTS, NOT IN THE SHARED PRIMITIVE:
  `len(images) != 4 -> refuse` landed at `fold_amplification`,
  `fold_ppgo_correction` (falls back to raw ppGO) and
  `channels.born_carrier_from_partition`; `_merging_fold_pair` was left
  alone because its 5 consumers include a `_pearcey_cusp` disjunct that
  tightening would silently flip. Generalizes the earlier
  `_is_interior = len(images) >= 4` fix: REAL-IMAGE COUNT is the exact
  interior/exterior discriminator for both parities. Exterior positive
  parity = 2 real images and the merging pair there is COMPLEX, so
  `_merging_fold_pair` returns the FAR pair and the Airy correction is
  spurious. Interior 4-image paths stay byte-identical — a count guard is a
  strict no-op where the count already matches.
- BREAK AN IMPORT CYCLE BY HOISTING THE CONSTANT DOWNWARD: `operator`
  cannot import `channels` (channels imports operator), so the shared ghost
  gates `_GHOST_SEPARATION_MIN`=0.7 and `_GHOST_DECAY_IM_THRESHOLD`=0.4
  were hoisted into `geometry` (foundational: stdlib + numba/numpy/scipy
  only) and BOTH consumers bind from there. Preserve and assert any
  derivation invariant the old site encoded (0.4 == _FARFIELD_WINDOW_RADIANS
  / 5.0).
- ABSOLUTE-FRAME vs t_min-FRAME CARRIER: an arm-ladder rung serves in the
  ABSOLUTE frame — call `geometry.ghost_kernel` directly and add
  `kernel * cmath.exp(1j*w*delay)` with a `+` sign and a NON-conjugated
  tau_c. `channels.farfield_ghost_term` is the min-subtracted t_min-frame
  variant and must never be reused for absolute-frame serving. Sign check:
  Im(tau_c) > 0 non-conjugated gives |carrier| = exp(-w Im tau_c) DECAY;
  the conjugate blows up.
- EXCEPT ORDER ENCODES DECLINE vs REFUSE: catch specific-first —
  `GhostAbsentError` (interior 4-image) is a DECLINE (fall through; interior
  serve stays byte-identical), bare `GhostDomainError` is a REFUSE (return
  None, never a zero ghost), `LensDomainError` last. The hierarchy is
  GhostAbsentError < GhostDomainError < LensDomainError, so a reversed order
  silently collapses three meanings into one.
- A HANDOFF IS NOT A FIX: when the Inspector re-raises the SAME finding
  because a routed fix was written to a handoff file but never applied,
  break the loop by executing it yourself — permissible when the target is
  PRE-EXISTING tests asserting already-landed physics (not certification of
  your own new code) and the Inspector explicitly directs execution. Ground
  every fixture edit in FRESH execution (image counts, pair gaps, ladder
  values), never in the handoff's prose.
- A GUARD THAT LANDS NEW PHYSICS INVALIDATES FIXTURE PREMISES, NOT ONLY
  ASSERTIONS: after the fold refusal, fixtures labelled "off-axis interior"
  measured 2-image EXTERIOR, and one class's premise (an interior on-axis
  cusp source at gamma=0.5) was physically unsalvageable — no such source
  exists. The repair is to re-derive the fixture from live geometry or
  invert the test to the new truth, and to fix every docstring asserting the
  dead premise.

## 2026-08-14 builds (symmetry_tie_c3_admission)

- DOUBLE-MASK BUG PATTERN: `geom.images` from `geometry.find_images` is
  ALREADY the real-only array (length k, e.g. 2 for a saddle 2-image draw)
  — re-indexing it with the length-4 channel `real_mask`
  (`geom.images[real_mask]`) double-masks and raises IndexError on every
  2-image draw. Use `np.asarray(geom.images)` directly; `real_mask` is
  only for length-4 CHANNEL arrays (delays/etc from `_frame_delays`).
- SADDLE FAR-FIELD GATE REPLACED (`_saddle_farfield_analytic_serves`,
  likelihood.py): after an eta-floor (measured-boundary) attempt was tried
  and reverted, the shipped gate is a c3-LED CERTIFICATE: `est =
  ppgo_error_estimate(real_images, source, matrix, w_lo)`; `est is None ->
  refuse` (primary merge discriminator); admit iff min pairwise image
  separation >= `_SADDLE_FARFIELD_MIN_IMAGE_SEP`(0.05, defense-in-depth
  backstop) AND `_SADDLE_FARFIELD_SAFETY`(20.0)*est <=
  `_SADDLE_FARFIELD_CERT_BAR`(1e-3). Retired `_SADDLE_FARFIELD_RHO_FLOOR`
  and the eta-floor mechanism entirely.
- RETIRING A SUPERSEDED TEST SUITE (git rm, not deprecate-in-place): before
  deleting, confirm (a) zero refs to the retired constant tree-wide, (b) no
  module imports the file (leaf node), (c) any locally-scoped test helper
  duplicated by the replacement suite is safe to lose. All three orphaned
  old-gate suites here were `git rm` only after this check.

## 2026-08-14 build (born_residual_wiring first-class intercept, WP-F/WP-G/WP3)

- MIRROR-REUSE CHECK BEFORE WRITING A NEW MIRROR: when lifting a buried rung
  to a first-class intercept that reuses the SAME shared helper methods
  (_ppgo_cell_coords/_ppgo_band_split/_ppgo_cell_ceiling here), the existing
  census/training mirror that already inlines those same methods needs NO
  arithmetic change — verify equivalence logically (De Morgan on the
  boundary conditions), then add WHY comments instead of a second mirror.
  Confirms 2026-08-13's MIRROR FIDELITY rule from the other direction: a
  mirror can already be current even across a reachability-lifting build.
- SENTINEL AUTO-ATTACH PATTERN: an optional collaborator (BornResidualChart)
  attached at construction via a private sentinel default
  (`_AUTO_BORN_CHART`) — load() raising refuses loudly, but a load ANOMALY
  falls back to None + RuntimeWarning (pure-engine behavior preserved);
  explicit `chart=None` opts out cleanly. Distinguishes "never configured"
  (silent None, by design) from "configured but broken" (loud warning).
- BAND-SPLIT ZERO-ABOVE-w_trust IDENTITY: below_mask=(dense_w<=w_trust) or
  all-True when w_trust is None; chart_w=dense_w[below_mask] carries
  IDENTICAL float64 values to the unsplit call, so a served/unsplit
  byte-identity pin can be built purely from the mask logic without an
  engine call — reusable whenever validating a new band-split consumer.


## 2026-08-14 build (saddle rho<1 per-cell relaxation, WP1)

- BLANKET GUARD -> PER-CELL ALLOWLIST: replaced a blanket saddle-rho<1
  UNKNOWN guard with `_SADDLE_RHO_RELAXED_CELLS`, keyed on EXACT float64
  gamma-edge equality (e.g. `[1.1572945272629378, 1.3393306228327468]`) —
  gate placed AFTER the value computation so refused cells still compute
  before being discarded (cheap, keeps the code path uniform). w_ceiling
  gained a matching consistency gate. Duplicate SITE1/SITE4 guards in
  likelihood.py/surrogate_census.py deleted in favor of the single
  CertifiedPpgoMap-owned allowlist (map methods are the source of truth,
  consumers delegate rather than re-guard).
- EXACT FLOAT EDGES MUST COME FROM THE SHIPPED ARTIFACT, NOT A HANDOFF
  PROSE VALUE: an allowlist keyed on float64 equality needs the true
  `repr()` read from `CertifiedPpgoMap.load()`'s grid, never a rounded
  value copied from a driver/inspector handoff note.

## 2026-08-14/15 build (F081 saddle tube fundamental training + tiling_census)
- ENGINE-FREE IS NO-CALL, NOT NO-IMPORT: importing any module inside
  cogwheel/lensing/ necessarily runs the package's __init__ chain (prior/
  posterior/marginalized_likelihood -> chang_refsdal -> channels ->
  _schwinger), so amplitude-engine module OBJECTS load at import time
  regardless. The achievable + load-bearing guarantee is NO ENGINE CALL +
  mpmath never entering sys.modules — verify via mock.patch booby-traps on
  the actual evaluate entry points, not via namespace-absence checks alone.
- HETEROGENEOUS PER-ARC SCALAR NEEDS BOTH EXTREMES COMPUTED: when a shared
  admission scalar (e.g. eta_max/exclusion_rho) is derived from a per-arc
  quantity (arc_r_min) that varies widely across arcs on the same band
  (~23x anisotropy measured), compute BOTH max_eta_max and min_eta_max and
  route each existing consumer to whichever currency it actually needs —
  don't silently reuse one extreme everywhere out of convenience.
- REMOVED DATACLASS FIELD FAILS ONLY AT COLLECTION TIME: a config literal
  referencing a since-removed field (e.g. TrainingConfig(max_tube_arcs=4))
  inside a test CLASS BODY only raises TypeError when the class body
  executes (pytest --collect-only or actual import) — py_compile does NOT
  execute class bodies and reports clean. When told to fix a removed-field
  break, verify with collect-only, not py_compile alone.
- A NON-VACUOUS FIX RE-DERIVES, NEVER JUST DELETES/LOOSENS: fixing broken
  tests after a signature/field removal by re-deriving the expected value
  from the live production selector (e.g. calling _tube_training_arcs
  directly) is the correct fix; deleting the assertion or loosening the
  tolerance is not — Inspector's pass-2 explicitly required this pattern.


## 2026-08-15 build (lobe_cusp_axis_edge_tolerance, WP1)

- WP1 edge-coincidence tolerance in `_lobe_cusp_axis_map` (surrogate.py):
  relaxed the two strict cusp-vs-edge guards (`if not cusp_angle > theta_hi:
  raise` / `< theta_lo`) to admit a cusp coincident with the side-appropriate
  edge within `_CUSP_EDGE_COINCIDENCE_ULPS = 8` ULPs (tol =
  ULPS*eps*max(1,|edge|,|cusp|)); d at that edge clamped to 0 via
  `max(..., 0.0)`. KEEP-MAP semantics (return type stays non-Optional tuple)
  chosen over the sibling `_deltoid_cusp_axis_map`'s Optional-return pattern,
  specifically to avoid the known latent `_chart_from_npz` unconditional
  `data['theta_to_u']` KeyError trap (documented in the 2026-08-08
  lobe_cusp_coordinate entry above) — a None-returning fix would have hit
  that trap on NPZ round-trip. Genuine interior straddle still raises
  ValueError. Const name ends `_ULPS`, not in the part0 absorber regex
  suffix list (_EPS|_MARGIN|_FRAC|_STANDOFF|_SAFETY) — flag for a future
  Tidier/Librarian naming-convention sweep if that allowlist gets extended.
- SIBLING-SHAPE AUDIT REUSABLE RESULT: of the three cusp-axis-map siblings,
  only `_lobe_cusp_axis_map` had a strict-inequality-at-coincidence shape.
  `_wedge_cusp_axis_map` pins its cusp to the domain boundary by
  construction (origin='low'/'high') so it has no cusp-vs-edge guard at all.
  `_deltoid_cusp_axis_map` already used non-strict `<=` branch selection
  plus straddle->None, so it was already safe. When auditing a sibling
  family for the same defect shape, check the GUARD STRICTNESS, not just
  whether the function has a similarly-named guard.

## 2026-08-15/17 build (serve_route_census)
- CensusConfig-style duck typing: `draw_samples(config)` only reads
  `config.n_samples`/`config.seed` — any frozen dataclass with those two
  fields works as its argument; don't couple a new census's config type to
  `surrogate_census.CensusConfig`.
- `cancellation_exponent` RAISES LensDomainError for gamma>=1 (i.e.
  1-kappa<=|gamma|) — callers must gate saddle-parity draws (pass math.inf
  directly, mirroring operator._saddle_grid) rather than calling it
  unconditionally.
- CAUTION: don't trust an in-session "X does not exist in module Y" claim
  without a fresh grep — CancellationError DOES exist in cogwheel/lensing
  (operator.py's F_op raises it, IS-A RuntimeError, per this same file's
  earlier entry); a build note asserting it "does not exist in
  cogwheel/lensing" was investigating one call site's needs, not the whole
  package — verify absence claims package-wide before recording as fact.

## 2026-08-17 build (tube_beat_free_representation, multi-launch recovery)
- MODULE-SCOPE FORWARD-REFERENCE CAVEAT (refines the SINGLE-SOURCE A
  CONVENTION note above): "forward refs resolve at call time — fine"
  is true INSIDE a function body, but NOT for a call at MODULE SCOPE
  (e.g. a test file's top-level constant built by calling a helper
  defined later in the same module). py_compile/AST parsing won't
  catch the ordering problem; pytest collection raises NameError and
  aborts the whole file/suite. Move the def above the module-scope
  call site, or make the constant lazy.
- DUAL-USE GATE DECOUPLING: when a production admission/serve gate is
  also probed by a purely STRUCTURAL/diagnostic question (e.g. a census
  classifying a synthetic, not-necessarily-buildable source), don't
  force the diagnostic caller to satisfy the full production gate —
  add a keyword-only override (default preserves production behavior
  byte-identically) so the diagnostic can bypass the orthogonal
  buildability check while still exercising the structural logic under
  test.
- DRY SINGLE-SOURCE MOVE VERIFICATION: when consolidating a duplicated
  helper down to one canonical module and re-importing it elsewhere,
  verify the move with an object-identity check (`a._helper is
  b._helper`) at runtime, not just "import succeeds" — identity proves
  there is truly one implementation, not two copies that happen to
  agree today.


## 2026-08-17/18 build (low_w_diffractive_rung, WP1-WP3)
- `point_mass_g_derivatives` (_hyp1f1) ALREADY bakes prefactor_c (the exact
  C(w), w*ln(w) phase) into every returned kernel value via its internal
  `_carrier` — do not re-multiply by prefactor_c on top of it, or the phase
  double-counts. Check this whenever composing a new analytic object from
  point-mass kernel derivatives.
- HONEST SELF-CONSISTENCY GATE (generalizes the gate-bounds-wrong-object
  lineage to a certificate being shipped for the first time): after
  computing a closed-form candidate boundary (e.g. w_low), re-evaluate the
  ACTUAL leading omitted series term AT that boundary (the worst-case point
  in the served band) and compare its magnitude to the real certification
  bar; return None (refuse) if it fails — even when the closed form "looks"
  principled (derived from a reference-frequency-held estimate), since such
  estimates can have the wrong monotonicity vs the true w-dependent tail.
- ERROR-METRIC CURRENCY instance: honest_error must normalize by the TOTAL
  amplitude-space magnitude `lam*sqrt_mu` (lam=1-kappa), never bare
  sqrt_mu, whenever kappa can be nonzero — bare sqrt_mu understates the
  relative error by a factor of lam.
- Reused split-mask helper across an INVERTED-polarity rung (ceiling rung:
  engine populates BELOW the split, fold populates ABOVE — opposite of the
  usual below-trusted convention): the shared helper's null/inactive-split
  fallback (all-True below_mask) is only safe for the "below is trusted"
  polarity. For the inverted rung, explicitly force below_mask to all-False
  and skip the engine call when the split is inactive, or every above-split
  node gets silently routed into the wrong populator (regression vs HEAD).

## 2026-08-18 build (born_farfield_completion, WP1/WP2)
- SPEC GUARD UNREACHABILITY: a spec's stated post-call guard (`if x == 0:
  return inf`) can be unreachable in practice if the callee it wraps already
  raises (e.g. ValueError via math.log on the same zero condition) before
  returning -- wrap the call in the matching except clause and keep the
  post-call check only as defense-in-depth, don't rely on it as the sole
  guard.
- GATE-LIFTING ORDERING: lifting a refusal gate to attempt a certificate-
  gated serve for previously-refused queries can require solving geometry
  that the old refusal returned before touching (pre-geom None) -- a
  propagating exception (e.g. LensDomainError) from that now-reached geom
  solve is outcome-preserving, not a regression, if the engine path below
  would raise the identical error for the same geometry.

## 2026-08-18/19 build (tiling_plan campaign module + F083 arc-trim promotion)
- New engine-free demand-sized module pattern (tiling_plan.py, mirrors
  tiling_census.py): lazy `_load_production_modules()`, never import
  surrogate_census (pulls the engine at module load). Cost currency
  SECONDS_PER_CALL=0.0903 reconciled against tiling_census's
  _SECONDS_PER_LABEL=0.09 in an emitted note (deliberate ~0.3% gap, not a
  DRY defect); total_calls = total_nodes * _LABELS_PER_NODE(8), single-
  sourced from tiling_census. Escalation verdict is RECORDED
  (should_escalate + reasons), never raised.
- w-axis DD-ceiling clip bug pattern: an axis helper that reports a
  MEASURED upper edge from demand records must also clip it to any
  documented hard ceiling (here the DD engine ceiling, single-sourced via
  census header `w_band_edges['w_ceiling_dd']`, propagated through every
  call layer as a trailing `Optional[float]=None` kwarg with a lazy
  fallback resolver) — clip BOTH the "records exist" branch and the
  "empty records -> prior-box fallback" branch, and emit a distinguishing
  source tag for each (e.g. 'measured_clipped_dd' /
  'prior_box_fallback_clipped_dd') so downstream consumers/tests can tell
  a clipped edge from a genuinely-measured one.
- Adding a new trailing optional kwarg with a lazy-resolving default is the
  standard backward-compat move when threading a new cross-cutting value
  (ceiling, tolerance, etc.) through several existing helper layers whose
  callers (esp. test fixtures) are all positional and pre-date the change.
- Promoting a test-fixture algorithm into production (F083 tube arc-trim ->
  `surrogate_training._trim_tube_arc`, keyword-only args): copy the tuned
  scan verbatim (constants + point count), gate strictly on the parity
  argument FIRST (`if parity != 1: return arc` — byte-identical passthrough
  for the ungated case), and if the trimmed value feeds a closure default
  bound at def-time (e.g. `arc=arc` in a nested builder), reassign the
  loop-local variable BEFORE that closure is defined so the trim actually
  propagates — a trim computed after the closure captures the old value
  silently.

## 2026-08-19/20 (diffractive_certificate_fit lineage, w_low_fit implementation)

- PASTE-THEN-MEASURE CIRCULARITY: a calibration script's margin report must
  use the SCRIPT's fresh fit (derate*_evaluate_fit), NEVER the production
  module's already-baked w_low_fit — the latter re-validates the PREVIOUS
  paste (placeholder coefficients) and reports a stale bake as green.
- FENCE-BRANCH CATEGORICAL OVER-SERVE (fenced build): a fence branch that
  returns the hard ceiling for the deep interior is catastrophically wrong —
  the engine-honest ceiling inside the caustic is ~4-34, NOT the DD cap 60
  (rel error ~1e7 at gamma=0.5); serve the interior with the SAME fit as the
  exterior (it's calibrated on interior cells), decline only the shell
  [RHO_LO, 1+DELTA]. Wall collapse is structural: a negative
  log(1-gamma')^2 poly coeff forces P->-inf as gamma'->1.
- EXPLICIT GROUP SLICING OF A GROWING FEATURE VECTOR (corner build): once a
  fit gains a trailing feature group, slice each group by index
  (poly[:n]/harm[n:n+m]/caustic[n+m:]) — a bare `features[n_poly:]` zip
  silently DROPS trailing groups when the leading group size changes.
- ON-GRID-GREEN ≠ OFF-GRID-GREEN (aliasing pass2): an aliased re-bake passes
  on-grid yet under-serves ~1e4 off-grid (w_low_fit ~0.0014); verify a
  re-baked coefficient block off-grid before pasting. But FIRST re-check the
  calibration oracle's row coverage — pass3 proved the "degenerate fit" was
  a symptom of a kappa-oracle bug (12 kappa rows skipped -> 8 aliased
  thetas), not the representation itself.
- SHA-SKEW (pass3): a provenance SHA stamped from committed HEAD does NOT
  identify an UNTRACKED baking script's state — commit the script (with its
  fixes) BEFORE the run so the stamp is meaningful.
- CAUSTIC-POINT SIGN SENSITIVITY: geometry.caustic_point is
  gamma-sign-sensitive (effective_u goes negative for -gamma); use
  abs(gamma_prime) in any symmetric caustic-relative rho discriminator.
- BETA-ROTATION IDENTITY: y_eig = [[cosB,sinB],[-sinB,cosB]] @ (y/sqrt(lam))
  is exactly complex exp(-1j*beta) multiplication — verified equal; use
  whichever form keeps the surrounding code real/complex-native.

## 2026-08-20 (low_w_diffractive_chart build, WPs + INS-1-001/2-001/2-002 fixes)

- CONTENT-HASH COMPLETENESS CONTRACT (INS-2-002): a field stored in a
  hashed artifact (npz) must be folded into the content hash at EVERY site
  (train bake + load recompute) — never omit a correctness-critical field
  (e.g. declined_mask) "to avoid breaking a test helper"; the test helper is
  the intended contract, match production to it in the SAME pass (hash
  fields in IDENTICAL order; bool->float64 hashing is byte-deterministic).
  A missing field lets a tampered/all-False mask load silently and serve
  above the certification bar.
- CLOSURE-CAPTURED SCALAR REBIND (INS-2-001): in a dispatch function, don't
  rebind a closure-captured scalar gauge (e.g. rho) to a different-gauge
  value — use a fresh local (rho_dir) so the recorded field's gauge stays
  invariant. Here rebinding rho to the DIRECTIONAL _diffractive._caustic_rho
  (gauges differ ~1.45-6.2x) silently corrupted residual_demand's 3-way
  split buckets for every subsequent fall-through draw.
- TWO-SIDED SERVED-ERROR MARGIN + PER-CELL DECLINE MASK (INS-1-001): the
  margin report must measure |derate*r_interp - r_engine|/|r_engine| sup
  over the FULL grid (grid + off-grid theta midpoints), never raw
  interpolation error; derate = min(1.0, 1/max_overshoot) (NO hard 0.85
  amplitude cap — see architect_knowledge; a cap makes serves 15% low at
  exact-interp cells). Where served error (or the 1-derate uniform bias)
  exceeds CERTIFICATION_BAR, bake a per-cell DECLINE mask (3-D bool,
  D2-folded declined() lookup); serve returns None -> exact engine; census
  mirror reads the mask engine-free, declined covered draws stay
  engine_residual.
- SCALAR DE-RATE IS GRID-RESOLUTION-LIMITED (WP2 flag, NOT redesigned):
  derate = 1/max_overshoot is dominated by the single worst overshoot — a
  smoke w-grid (5 nodes) under-resolves the w-oscillation and crashes the
  de-rate to 0.05-0.15 (median ratio ~0.06-0.2); full bake (14 w nodes)
  expected toward 0.85, but the (1-derate) systematic magnitude bias is in
  TENSION with a two-sided "agree <= 1e-4" acceptance; the far-exterior
  wall band (rho~8, s~16) needs >>14 log-w nodes, so it may be
  interpolation-limited.
- RPURE ORACLE + NODE GRID (WP2): oracle r_pure = f_schwinger(w, y_eig,
  gp)*sqrt(1-gp^2)/prefactor_c(w); node source inverted from the fence
  discriminator (|y'| = rho*|caustic_point(gp,theta)|) — chart is pure
  reduced frame, no _reduced_shear/_rot_minus_beta. Full rho grid =
  [min(RHO_LO, 0.8*lo), max(RHO_HI, 1.1*hi)] from MEASURED wall-band
  2-image spread (measured [0.0775, 7.6115], seed 42) — never a literal;
  theta 16 full nodes (8 = harmonic Nyquist); log w [log 0.02, log 60].

## 2026-08-20/21 (low_w_diffractive_chart F_ref implementation + cusp-fallback WPs)

- ORDER-PRESERVING EXTRACTION IN A PERFORMANCE-CRITICAL DISPATCH (INS-1-001): extracting a shared bundle out of cusp_amplification must preserve the gate/early-return ORDER — an extraction that ran `_consult_pearcey` + cluster/far split BEFORE the ppGO fast rung + F074 error gate caused ~1000x slowdown AND a silent refusal (a primitive None short-circuits before the fast rung ever runs). Shipped structure: `_cusp_uniform_geometry` -> `_cusp_controls` (pure scalar, no quadrature) -> ppGO fast rung (early return) -> F074 error gate -> DEFERRED `_cusp_uniform_at_w` -> calibration -> total. The SPEC claim "returns before any table or quadrature lookup" is load-bearing prose; verify the CURRENT body before trusting it.
- ONE DATACLASS PER PIPELINE STAGE: `_CuspUniformForm` shrank to the 4 fields cusp_amplification actually consumes (uniform, far_sum, stationary_values, matched_delays); geometry and controls each live in their own dataclass (`_CuspUniformGeometry`, `_CuspControls`) so the bundle stays small.
- F_ref SINGLE-SOURCE AT SERVE: `_low_w_diffractive_chart_serve` rebuilds F_ref at serve via `reduced_source(gamma_prime,rho,theta)` + fold_cusp_reference(dense_w, ...), placed AFTER the rho computation but BEFORE chart.covers; None falls through to w_low_fit/engine. reduced_source needs no try/except when its caustic_point call is byte-identical to one already evaluated just above for the same args.
- covers() returns np.bool_ (not Python bool) in the w-given branch (grid elements are np.float64) — tests must not use `is False`.
- Trainer mechanics: `_fill_coefficients` returns (real, imag, n_refused, unbuildable_mask); off_served filled by BOOLEAN-INDEX assignment (off_served[~off_unbuildable] = served_errs[grid_n:]), never reshape; `_w_grid` uniform in w**(2/3): linspace(W_LO**(2/3), W_CEILING_SCHWINGER**(2/3), n)**(3/2); _SMOKE_N_W 5->8, _FULL_N_W 14->16.


## 2026-08-21 (low_w_shell_born_extension build — shell serve + gauge decouple)

- SILENT CUBIC EXTRAPOLATION BEYOND A GRID BOX (INS-1-002): chart.evaluate
  with bounds_error=False/fill_value=None will cubic-EXTRAPOLATE out-of-
  grid parameter axes (e.g. gamma_prime) silently — add an explicit grid
  box gate (`chart.gamma_prime_grid[0] <= gp <= chart.gamma_prime_grid[-1]`
  -> None) BEFORE evaluate whenever the serve keys on a parameter axis.
- CROSS-GAUGE SINGLE-SOURCE CATEGORY ERROR (INS-3-001): `_BORN_RHO_FLOOR =
  RHO_HI` looked like DRY but the two constants live in DIFFERENT rho
  gauges (Born: scalar-reach ppgo_map.caustic_rho; shell: directional
  _caustic_rho) — identical numeric value != identical physical surface.
  Decouple to an independent constant + honest comment, pin value equality
  only in a test.
- FULLY-RESOLVED BAND-SPLIT NULL FALLBACK (INS-1-002/005): the shared
  `_band_split_mask` all-True null fallback is WRONG when the whole band is
  resolved-below-trusted — add an explicit `w_shell <= w_lo -> None` guard
  so a fully-resolved below-trusted polarity declines rather than serving
  the whole band via the wrong side.
- SERVING CONVENTION: a propagating LensDomainError (e.g. from
  geometric_amplification inside a band-split branch) is outcome-preserving
  when the exact seed-engine path below raises the identical error for the
  same geometry — no try/except needed (per the _amplification_coefficients
  docstring convention).
- Shell trainer (scripts/train_low_w_shell_chart.py): grid gamma_prime in
  [GAMMA_LO, 1-DELTA_GAMMA_P], rho linspace(RHO_LO,RHO_HI), theta D2
  [0,pi/2], log_w geomspace(0.02,1.0); reduced_source single-sourced from
  low_w_shell_chart; NO derate/declined_mask (measured |r|~0.61-1.6);
  f_schwinger refusal -> SystemExit naming the node (fail-fast, never NaN);
  content-hash field order (gamma_prime_grid,rho_grid,theta_grid,log_w_grid,
  real,imag) IDENTICAL to load; Born trainer rho explicit [1.4..4.0] +
  w geomspace(0.4,60,13) (shell owns deep low-w at rho<=1.4), provenance
  driver_prerequisite = azimuthal sweep at rho=1.4 (N(theta)<=8 gate before
  re-bake). RHO_LO=_DIFFRACTIVE_FIT_FENCE_RHO_LO, RHO_HI=1.0+
  _DIFFRACTIVE_FIT_FENCE_DELTA imported from _diffractive (never re-typed).
