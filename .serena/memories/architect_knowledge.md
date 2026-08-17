# Architect Long-Term Knowledge

- If the source task/prompt is unreadable in-session, do NOT fabricate a
  task — emit an empty-WP plan flagging the blocker after >1 independent
  read attempt.
- Docs-only tasks: check whether the target already satisfies the goal;
  plan the minimal diff, not a rewrite.
- Simplifier verdict pattern: leanest correct response; no no-op
  verification WPs "just in case". A verify-only Coder WP (confirm
  invariants readable from bodies) is bureaucratic relay = Inspector's
  job; trim it. Timing/warm-cost measurements are diagnostics, never gates.
  A design element explicitly pinned by an upstream domain-expert ruling
  (e.g. Professor) is not open for a Simplifier alternative — phrase the
  WP directly around the pinned design (Build 8h-b).
- Verify any "code-pinned"/"already exists" claim with fresh find_symbol/
  grep BEFORE planning WPs on it; agents refusing to fabricate a missing
  primitive = plan failure, not agent failure.
- ROUTING (recurring, 5x+): doc-sync/SPEC findings -> post-gate Librarian
  with exact replacement text (SPEC.md in files_affected is informational);
  findings confined to test-file fixtures/constants/categories -> Test
  Developer, never Coder. When a WP is deliberately redirected off the
  default agent, NAME the executing agent inside coder_instructions — an
  implicit redirect silently mis-routes back to the default.
- Inspector SPEC-staleness findings caused by a completed build (not a
  Coder defect) should be OVERRIDDEN as Librarian-routed at triage rather
  than surfaced as Coder work; include the exact suggested replacement text
  in the override note so Librarian can apply it without re-analyzing.
  RECURRING (2026-08-07): also applies when the finding cites BOTH
  DATA_CONTRACTS.yaml AND SPEC.md describing a superseded schema after an
  approved WP shipped the new one — doc-sync is the automatic post-build
  Librarian phase's job, not a Coder WP, regardless of how many canonical
  surfaces are named.
- Don't escalate a perf/accuracy floor measured on a defective build —
  fix, retune, re-measure, then escalate. Unreachable target: document the
  measured floor honestly, escalate; never widen tolerances. Timing
  acceptance = machine-independent structural gates first; absolute ms
  ceilings arithmetic-derived, never machine-calibrated.
- Interpolation node budgets scale with oscillation content (cycles in
  band), not kink count; ship the cheap structural lever before a surrogate
  table; have the Professor sanity-check brief perf arithmetic.
- Batching: hoist grid-independent quantities; any accumulation-order change
  needs re-certification vs an independent oracle + solo-vs-batch
  certify-XOR-refuse identity. Scalar API = thin wrapper over batched core.
- Known FINDINGS bug patterns recur in sibling paths — grep before
  inventing a new mechanism.
- SINGLE-SOURCE A CONVENTION (this repo's #1 recurring bug class): the same
  rule re-expressed at N sites always drifts — min-subtracted delay frame
  lived at 4 sites; physical-vs-apparent `d_luminosity` disagreed across
  SPEC.md and two module docstrings; a gate keyed on "whichever array the
  caller passed" produced train/serve skew. Require every stage (tiler /
  chart / serve / census) to call ONE identical shared function, and carry
  values derived from a fixed input pair (e.g. (source,matrix) -> images,
  t_min, real positions) ON the partition/dataclass instead of re-deriving
  them inside hot-path functions. A new rung must reuse the existing tested
  primitive rather than create convention site N+1.
- Refusal boundary: thin Posterior-subclass override maps named domain/
  cancellation refusals -> -inf + metadata; raw likelihood keeps its raise
  contract; the sampler never catches (Build 4).
- Posterior requires prior.standard_params == likelihood.params EXACTLY;
  fix unmeasurable params via FixedPrior rather than omitting them. Reuse an
  existing prior with a documented option-deferral rather than block a build
  on the ideal coordinate (d_app deferred to Build 5).
- Cache determinism: snap proposals to module-constant lattices so a
  fiducial/cache entry is a pure function of the candidate.
- Prefer one-line guards + fallback-to-certified-direct over topology-aware
  partitioning (Simplifier trim). Plan a __getstate__ dropping derived
  caches (bases define none); JSONMixin/get_init_dict path unaffected.
- Extending a byte-frozen validated path to a new regime/parity: add
  SEPARATE parallel functions behind a classification gate that mirrors the
  frozen path's gate; never refactor the frozen one. Shared entry points get
  an optional flag so the default call stays byte-identical; keep the
  regime-branch decision INSIDE the new function (Build 6 saddle). Same for
  a genuinely new physical contribution (e.g. a ghost/complex-saddle term):
  build a DEDICATED kernel, never route through unrelated existing kernel/
  delay/index helpers, plus explicit degenerate-axis refusal (Build 8h-b2).
  REFINEMENT (Build saddle Born carrier): this "separate function" rule is
  for a genuinely NEW physical contribution. When instead mirroring an
  EXISTING rung to the opposite-sign/mirror-symmetric regime of the SAME
  physical object (e.g. real vs complex Morse index by parity sign), an
  IN-FUNCTION branch keyed on the same classification gate is correct and
  cheaper; generalize an existing one-sided wall-margin guard to two-sided
  when the new branch admits the opposite side. Known exact phase/index
  constants (e.g. Morse factor i^n) must be hardcoded as the exact literal,
  never evaluated via cmath.exp/trig — sub-eps round-off in the
  transcendental form can break a downstream flat-magnitude pin invariant.
- GATE CURRENCY: prefer a STATE-INDEPENDENT (geometric) admission currency
  over a state-dependent one whenever train and serve can see different
  states. The w-dependent decay gate (w_min * Im tau_c) skewed train vs
  serve because each keyed on its own w-grid; re-keying to a bare geometric
  separation (min over real images of |x_a - x_c|, Einstein units, no
  normalization, no floor) killed the skew BY CONSTRUCTION. Do NOT add a
  secondary "w >= const" guard afterwards — that resurrects the skew
  (supersedes the 8h-b w*Im tau_c currency ruling; Build 8h-d1).
- Pin a new threshold constant INSIDE the measured refuse/admit gap, biased
  toward the conservative side (false-admit = silent lnL bias; false-refuse
  only falls back to the exact engine), and write the re-key rule into the
  brief (e.g. geometric mean of measured refuse-max/admit-min) so a
  contradicting verify sweep has a defined response.
- Acceptance for a subtractive/corrective term = a config-agnostic
  DO-NOTHING CONTROL (residual WITH the term <= residual WITHOUT it, on
  EVERY admitted config), not a per-config tolerance table — that is
  exactly the property a badly-keyed gate fails.
- A new ANALYTIC rung must be expanded about the true leading order (e.g.
  sqrt(mu_macro)), never about 1 wherever the background is non-trivial,
  and its small-parameter power direction expert-checked (the correction
  must vanish in the asymptotic limit — an inverted power passes dimensional
  checks while inverting the domain of validity). Its gate is analytic
  (term-estimate + margin to the convergence wall), not Coder-measured.
- Surrogate/emulator design (Build 8a): emulate the SMOOTH symmetry-
  invariant object (the beat-free envelope E(w)), NOT the oscillatory total;
  build ONE interpolant PER topology region (parity/image-count) since the
  decomposition changes topology at caustics; exact-engine fallback near
  caustics + outside the box. Reduce out any EXACTLY symmetry-eliminable
  parameter (beta via eigenframe rotation) BEFORE training.
- Conservative-serve gate = axis-aligned box containment + exclusion balls
  around refused points + per-sample refusal propagation; NEVER a learned
  mask (a false negative is a correctness bug, not an efficiency miss).
  Default surrogate=None -> exact path byte-identical.
- When a GLOBAL tolerance tightening blows the certified hot-path timing
  gate (measured, at plan gate), reject it and re-key the constant on a PURE
  fn of the candidate params (gamma'-keyed LOO stop): certified fast region
  stays byte-identical and cache purity holds; tighten only the sub-region
  that needs it.
- One uniform prior can span two physical regimes when the regime is a
  deterministic fn of a sampled coord (parity from gamma) — no discrete
  label, no sub-prior; the boundary is a measure-zero named refusal -> -inf
  at posterior, never prior special-casing.
- Two-tier verify: in-build = small reduced-domain surrogate/fixture + fast
  falsifiable gates; full-box training/census/PP-plots are POST-BUILD driver
  steps named in acceptance, never in-build test specs.
- Accuracy/eps gates evaluated at artifact-build time must persist their
  metric in per-artifact provenance so a reload/reuse path re-applies the
  same gate, not just the build path.
- Distinguish a handoff/switching exponent (asymptotic-regime boundary)
  from an accuracy floor before proposing to raise a ceiling constant.
- Feed Inspector-authored fix snippets through Simplifier before endorsing
  verbatim — a shape mismatch (e.g. dict vs flat-list) can ship a fix that
  passes in isolation but breaks the existing consumer contract.
- Grid/node reprovisioning: reuse an existing normalized held-out-error
  metric (e.g. LOO) to decide how many nodes to keep/drop rather than
  hardcoding a reduction heuristic — let a probe decide (Build 8h-b).
- When an accuracy/interior label becomes ill-conditioned in a parameter
  sub-region (e.g. near a higher-order catastrophe), switch that sub-region
  to an alternate ALREADY-ESTABLISHED label/envelope with a concrete
  falsifiable pass/fail pair, rather than tuning the ill-conditioned label.
- New accessors added to an existing family (e.g. w_ceiling alongside
  w_cert/w_trust) should mirror that family's naming/behavior exactly
  rather than invent a new sentinel type (Build 8h-b).
- A WP framed as "find a bug that a test might expose" with no pre-
  identifiable defect is not a valid Coder WP — it's a forbidden measure-
  and-decide campaign; a repair already committed but unexercised by
  tests is a Test-Dev completion/port task, not Coder (Build 8h-b5).
- When two sibling code paths (e.g. interior/exterior admission) derive
  from the same shared geometric anchor (e.g. caustic cusp rays),
  symmetrize a structural fix across BOTH rather than patching one — an
  asymmetric fix inherits the same kink the other side already solved.
  A gate relaxation that is only sound GIVEN a prerequisite structural fix
  must be planned as ONE merged WP with it; landing it alone is unsound
  (Build 8h-b6).
- NAMING HAZARD: don't overload an established term when adding a regime
  (here "far-field" = a trained chart a GAUGE, NOT weak deflection) — pick a
  distinct name in the brief or the WPs inherit the ambiguity.
- FRAME-INVARIANT RELABELING: when a trained/interpolated label carries a
  per-node reference-frame phase (e.g. min-relative time delay) that varies
  node-to-node, the fix is to relabel into an ABSOLUTE frame (multiply out
  the node-dependent carrier) before interpolation, and hand the frame value
  back at reconstruct time to de-tilt — never just tighten a continuity
  guard around the frame-mixed label. Pair with an axis-schema version bump
  (hard-refuse pre-relabel artifacts) and a dedicated carrier-continuity
  guard on the NEW label (Build 8h-d2).
- Exact/closed-form geometric nodes (e.g. astroid cusp angles) that are
  independent of the varying parameter should be UNIONED into the
  interpolation grid as exact spline nodes rather than left to a uniform
  grid to approximate — gate the union on the regime where the closed form
  actually holds (e.g. positive parity only), and derive it from existing
  geometry primitives directly rather than importing a sibling module (risk
  of circular import).
- Two WPs that both edit the SAME function/entry point (e.g. both touch
  `from_engine`) must be sequenced via depends_on, never planned to run in
  parallel — even when their changes are conceptually orthogonal, they will
  conflict on the same code region.
- TWIN-GATE root acceptance: gate a numerically-located root on TWO
  independent checks — a sign-crossing test on the target derivative AND a
  magnitude/scale test against a locally-measured off-target scale — before
  serving it; a single-condition accept risks aliasing near a divergence
  (Build 1c analytic cusp vertex).
- Domain-necessary refusal at a divergence: when a legacy numerical scan
  can alias across a genuine divergence and serve a finite-but-meaningless
  value, an analytic replacement's correct behavior is a NAMED refusal
  (None) at that boundary, not chasing/snapping through it — a "pure
  uniform" simplification is wrong wherever a domain refusal is physically
  required (e.g. a macro-saddle diverging wedge edge).
- Served-values acceptance for a serve-path swap: PRIMARY = perturb the
  served input by measured/physical increments and assert the served
  output moves less than the accuracy bar (insensitivity, not bug-for-bug
  equivalence with the old code); SECONDARY = old implementation as a
  comparison oracle, with configs where the old code was itself buggy
  explicitly carved out and documented as an intended improvement, not a
  regression.
- A scalar magnitude guard on a proxy quantity can conflate an unrelated
  property (e.g. fold transversality) with the one it's meant to gate (e.g.
  cusp proximity) — same bug class as the retired _PROBE_ETA; prefer an
  exact-zero/sign tripwire over a magnitude filter when the magnitude isn't
  the actual physical quantity being gated.
- WP split when a fix spans BOTH production+its own docstring AND test-file
  helpers/prose: one merged Coder WP (prod+docstring together, never split
  a symbol from its own docstring) + one Test Dev WP (all test authoring +
  test-file helper/prose fixes, routed via domain_test_descriptions) —
  Coder never touches test files (Build 1d).
- When a brief swaps a finite-diff derivative for an analytic closed form,
  make orientation/SIGN agreement — not just magnitude — an explicit
  load-bearing gate: an independent golden sign table cross-checked
  against a non-circular construction (e.g. an image census), never mere
  self-consistency of the same function against itself (F041, Build 1d).
- PURE PORT builds (0 numerical changes): assign ONE domain_test_description
  per owned file rather than aggregating, to dodge F057 cross-suite budget
  blowup from a single file importing multiple heavy suites; a VALUE failure
  in a pure port is a real finding, STOP.
- Near-wall oracle for a caustic-reach scan MUST use the parametric caustic
  radius r(u) = |y(u)| (F026 bracket-refine), NOT a source-plane ring sweep
  — a ring misses the thin near-wall spike and gives a systematically low
  oracle.
- INTERIOR WEDGE CHART DESIGN (Build interior_wedge_chart): caustic-relative
  (r, theta_wedge) coordinates for 4-image interior sources; D₂ symmetry
  (Klein four-group); from_wedge_engine entry; kind='wedge' NPZ. See
  coder_knowledge for full implementation checklist.
- DD PRODUCT BOTTLENECK: far-field charts capped at w <= ~60 (double-double
  precision limit). Fix direction: r-dependent w ceiling; ppGO serves below
  w_cert. See coder_knowledge for cap formula.
- CENSUS BAND-SPLIT CONSISTENCY: `characterize_sample` must replicate the
  SAME band-split logic as the production serve path; share the identical
  band-edge accessor between census and serve.
- Arc-length reparametrization design: use rep_gamma = median(gamma_grid)
  to minimize worst-case effective excursion; a single-gamma map is adequate
  for topology-stable bands.
- D-NORM-EVAL DESIGN (Build d-norm-eval): implement d/R_c normalization as
  IN-MEMORY-ONLY opt-in (d_normalized flag on from_engine; per-(gamma,s)
  rc_table stored on chart; one divide in serve transform + box gate). NO
  NPZ persistence/schema bump in the eval build (deferred to promotion build
  — OWED). R_c MUST be per-(gamma-node, s-node); a single-theta R_c repeats
  the arc-map mistake. A/B test = MEASUREMENT not decision-oracle: gate hard
  correctness invariants (round-trip bijection, min R_c>floor, train/serve
  box parity, node-exact on stored grid, default byte-identical,
  eps_norm<=1.1*eps_raw). Record stratified near-wall/far-tail eps for
  driver's >=2x promotion gate (never gate in-build on this).
- GEOMETRIC COVERAGE SCRIPT PATTERN (scripts/measure_far_zone_crossover.py):
  measure tube+far-field tiling coverage at the C8 boundary using
  `_coordinate_radius_bounds`, `_min_curvature_radius`, `_astroid_arcs`,
  `_saddle_arcs` from surrogate_training and geometry.nearest_caustic_point
  for exact distance. exclusion_rho = 1.0 + (reach_max + eta_max_max) -
  coord_radius_min (matches production formula exactly).
- FOLD-PPGO INTERIOR HANDOFF DESIGN (Build ppgo_interior_handoff): dual-gate
  serve path in `_surrogate_coefficients` for rho<1 draws above InteriorWedge
  Chart w-ceiling. Coarse gate: ξ_min >= 4.0 (Chester-Friedman-Ursell param,
  ensures ~2 full Airy oscillations). Fine gate: c_A * ξ^{-3/2} < CERT_BAR
  (1e-4) — correctly admits only at large w (e.g. census xi~528). DO-NOTHING
  property: fold_ppgo_correction works on ALL regimes (calls geometric_ampli
  fication for all 4 images), so it can never be WORSE than raw ppGO. Census
  category 'ppgo_fold' with served=True skips fallthrough bucket. Mirror
  _XI_FOLD_THRESHOLD as a LOCAL constant in census.py (same pattern as
  DD_PRODUCT_MARGIN). One WP pattern: likelihood.py + surrogate_census.py.
- GENERIC BOUNDED-RECURSION SUBDIVIDER DESIGN (Build
  subdivision_recursion_and_coordinate_cleanup, 2026-08-07): when two
  near-duplicate tile subdividers exist (e.g. far-field + wedge, ~200 lines
  each), unify into ONE generic helper parameterized by (child-box splitter,
  build callable, gate/admission fn) plus a small hard-coded recursion depth
  cap (e.g. 3); keep the two original named functions as THIN WRAPPERS that
  build closures over the module's own chart-builder/probe helpers and
  preserve their exact signatures + all call sites. Require a BYTE-IDENTITY
  pin for the common case (a tile whose children all pass at depth 1 -> no
  recursion -> only ADDITIVE report keys, e.g. achieved_depth, change).
  A structural-refusal branch (e.g. CarrierDiscontinuityError) must never be
  recursed. Recursion can legitimately change a summary field's SEMANTICS
  (e.g. 'packed' becomes a full-subtree count instead of single-level) —
  this is an intended, brief-endorsed accuracy improvement, not a defect.
- CLOSED-FORM ROOT-FINDER REPLACES DENSE SCAN (r_caustic, same build):
  replacing a legacy dense numerical scan with a parametric-curve brentq
  bracket+refine (bracket count/parity keyed to topology, e.g. 48 nodes for
  astroid vs 720 for saddle) gives an order-of-magnitude speedup with no
  serve-value drift; the domain-necessary refusal at a true divergence
  (e.g. gamma=0, parity boundary) must raise the NAMED domain error
  directly from inside the new root-finder — never let a raw
  ZeroDivisionError leak from the arithmetic.
- SEMANTIC FIELD RENAME + SCHEMA BUMP PATTERN (WP3, same build): renaming a
  stored field to reflect its TRUE semantics (e.g. arc-length `theta_to_s`
  actually stores a cusp-adapted angular coordinate `u = d**(2/3)`) requires,
  in one merged WP: (1) the rename across every producer/consumer/NPZ-key
  site, (2) a schema version bump that HARD-REFUSES pre-rename artifacts
  (no migration/fallback), and (3) DRY-ing any per-field validator into one
  core routine parameterized by ordinate name so sibling fields keeping the
  old semantics (e.g. Tube/Lobe/FarField's genuine arc-length `theta_to_s`)
  are provably unaffected.
- A GATE MUST BOUND THE ERROR OF THE OBJECT IT ADMITS (dominant defect class
  — four instances in one day, 2026-08-13: F069 the estimate decayed while
  the true error stayed flat; F070 the clamp licence keyed on the LABEL, not
  the served value; F074 the radius gate bounded the error of the object the
  rung REPLACED; F076 the resolution gate read the wrong image pair). When
  planning any admission gate, name the served quantity whose error it
  bounds, DERIVE the estimate from that object's own asymptotics (e.g. the
  c3 series term for a ppGO serve), and calibrate it against an F069-safe
  oracle. "Conservative in practice" is not evidence — a gate keyed on a
  different object is wrong even where it happens to pass.
- REAL-IMAGE COUNT IS THE ONLY CAUSTIC DISCRIMINATOR THAT SURVIVES THE
  EXTERIOR: outside a caustic the merging pair is COMPLEX, so any gate
  computing min-gap / xi / resolution / delta_tau over REAL images is
  structurally blind there (F073 xi_min, F075 fold pair, F076 delta_tau,
  saddle mirror pairs whose delta_tau is exactly 0). Use
  `real_mask.sum() == 4` (interior) vs `== 2` (exterior) — exact for BOTH
  parities, free (already on the partition), 0/2400 disagreements vs the
  closed-form caustic. Never plan a rho<=1 or origin-radius interior test.
- FREQUENCY-INDEPENDENT ADMISSION GATES FOR ANY RUNG WHOSE VALUES BECOME
  LABELS (Professor ruling 2026-08-13, confirmed by measurement): a
  w-dependent floor on a rung that trains tables re-opens the train/serve
  skew build 8h-d1 retired. Gate on configuration geometry only (e.g.
  Im(tau_c)>=0.4 and min|x_a-x_c|>=0.7 for the ghost rung); if accuracy
  still seems to need a w floor, the rung is mis-scoped, not under-gated.
- A PHYSICS FIX'S CONSUMER-SUITE SWEEP MUST BE COMPLETE BEFORE THE NEXT
  LAUNCH (2026-08-13, cost: two builds stranded at the tree gate): the F072
  guard fix missed 1 of 5 consumer suites; the F074 fix had fallout in
  suites outside the eight swept. The full tree gate IS the sweep — run it
  BEFORE launching the next build whenever production semantics changed,
  never as a post-plan discovery.


## Polar re-chart (2026-08-07)

- ExteriorPolarChart (rho, theta_c) replaces FarFieldChart.
- Polar coordinate is single-valued, no medial-axis degeneracy, no foot-tie.
- m_lens_range override enables single-stratum train() calls for probes.
- 4 test classes skipped: need fixture migration for polar chart (can be done incrementally).
- Envelope definition validation widened to _KNOWN_ENVELOPE_DEFINITIONS (farfield + interior union).
- Design rulings (Professor + Simplifier): polar is the correct exterior coordinate; chart the saddle exterior with KERNEL_SUM and stamp FARFIELD_KERNEL_SUM (NOT MINUS_GHOST); single class (ExteriorPolarChart), NO dual FarFieldChart/ExteriorPolarChart backward compat; cusp carve-out sized in y-units.
- Deletion scale: 0a31fcf removed ~1064 lines from surrogate.py (FarFieldChart, _FarFieldArcMap, _to/_from_farfield_smooth, _farfield_serves, _caustic_arclength_map + (s,d) schema constants); across the whole build ~1550 deletions / 500 insertions in the three lensing modules (surrogate.py, surrogate_training.py, surrogate_census.py). Stale (s,d) artifacts hard-refuse at load (no identity fallback).
- STRANDED-BUILD PATTERN (a WP that deletes a chart class AND migrates its tests is the fragile WP): the polar build stranded at WP-3 (coder-4 agent error; 4d59a6d is a SALVAGE commit — core architecture valid, deletion + test migration unfinished, tree gate not run). Driver completed it manually: 0a31fcf deletion cleanup, 72f4b84 test migration, 5859a78 stale FarFieldChart ref + (s,d) docstring fixes. Expect to plan a driver completion pass for delete-heavy WPs.
- Exterior recursion probe (post-strand, production gamma_band_halfwidth=0.04): 31 charts, 9/31 pass 1e-3; depth histogram {0:2, 1:6, 2:7, 3:16}; 13 depth-3 tiles STILL fail (eps 1.2e-3..3.6, hundreds-to-thousands x tolerance) — subdivision to the recursion cap does NOT fix a coordinate-level disease. Root tiles intrinsically bad (foot tie_ratio degeneracy); recursion paper-overs them. Confirms (s,d) is wrong for the exterior bulk (coordinate-level, not resolution-level).
- Wedge v3 single-stratum probe (FINAL): train(regions=('wedge_interior',), m_lens_range=(10,15.8)) → 10 charts, 9/9 valid eps pass the 5e-2 bar (2.0e-3..1.6e-2, median 6.0e-3). Earlier "NaN median / 19 charts" was TWO probe bugs, not a coordinate failure: (1) full-prior config (13 w-strata → ~130 tiles, wrong scale), (2) reading chart.provenance in-memory which LACKS heldout_eps after NPZ load (read the NPZ provenance).
- fix_tree_gate_hang (housekeeping, same day): conftest's 900s timeout was a NO-OP because pytest-timeout was NOT installed — a timeout guard is inert without its package. Fix: install pytest-timeout, pin COGWHEEL_TRAIN_TIER="" in build env (train-tier tests un-skip if the env var leaks), add _f_schwinger_mpmath sentinel guard to conftest.py.


## Lobe subdivision + ppGO above-ceiling (2026-08-08)

- LOBE SUBDIVIDER (saddle_forensics WP-1): lobe now covered by
  `_subdivide_lobe_tile`, a THIRD wrapper over the generic `_subdivide_tile`
  (after far-field + wedge), wired into `_train_band_charts` lobe branch
  (subdivision replaces immediate ladder_served_gap). D2 cusp carve-out
  RESOLVED as no-carve-out-needed: existing eta_max tube-shell nearest-
  distance test in `_SaddleLobeAdmission.admits` already rejects near-cusp
  tiles (cusp vertices are in caustic_cloud); the added
  `_LOBE_CUSP_EXCLUSION_DISTANCE=0.1` constant was RETIRED (deleted) by the
  follow-on cusp-adapted coordinate build (98c4e7f, 2026-08-08) — the eta_max
  tube-shell test alone excludes near-cusp tiles; the carve-out was always
  redundant. Simplifier: WPs independent; carve-out sited in
  admits (not the tiler) — but ultimately no carve-out was needed at all.
- PPGO ABOVE-CEILING ENGINE-INTERCEPT RUNG DESIGN (Build ppgo_above_ceiling,
  WP-4): ONE Coder WP intercepting `_amplification_coefficients` BEFORE the
  engine eval. Gate: w_max>150 AND w_lo*min_delta_tau>=RHO_END (4.0);
  whole-band serve via fold_ppgo_correction + reconstruct_farfield
  (FARFIELD_KERNEL_SUM). Explicitly NO band-split, NO census mirror, NO
  kappa/beta guards. Simplifier: engine-intercept (NOT the surrogate path —
  unreachable there), band-split trimmed, engine-intercept cleaner.
  Professor: error ~1e-2 @150, ~1e-3 @500, decreasing trend; RHO_END=4.0;
  all-image serve; boundary-continuity is the primary gate.
- Lobe normalized-radius disease (Professor, follow-on): r_deltoid->0 at
  deltoid cusps by |dtheta|^(1/3), the SAME power law as astroid) —
  subdivision alone does NOT fix the coordinate-level disease; the clean fix
  is a cusp-adapted u=d**(2/3) coordinate (wedge pattern) — SHIPPED
  2026-08-08 (98c4e7f). Design record: `mem:lobe_interior_chart`.

## Lobe cusp-adapted coordinate build (2026-08-08)

- QUOTA-DEATH SALVAGE PATTERN: the build died at inspector-17 (quota
  exhaustion) AFTER coder-16's fixes were committed but BEFORE the Inspector
  verified them; the code was salvaged as b18e6a8 and a FRESH Inspector
  audit (re-run suites + re-derive invariants, 149 pass) was required
  before the build could close. A quota-killed build does NOT close on its
  last green partial pass — salvage the commit, then re-audit from scratch.
  Distinct failure mode from the stranded-build pattern (agent error); both
  need a driver/Inspector completion pass.

## Exterior polar cusp-adapted u coordinate build (2026-08-08)

- DESIGN (Build exterior_polar_cusp_coordinate, 1a97bbd): ExteriorPolarChart
  gains an OPTIONAL cusp-adapted `u=d**(2/3)` axis (`theta_to_u`) — parity==1
  (astroid) tiles train with cusp-adapted u via `_wedge_cusp_axis_map`; the
  macro-saddle exterior (parity==-1) stays raw-theta (None). Schema tag bump
  'exterior_polar_rho_theta_c' -> 'exterior_polar_rho_u_v1' (old tag
  hard-refuses). Simplifier: 1 main Coder WP + 1 small sequential WP; origin
  derived from box_center; waist split at `_wedge_theta_waist`; retire the
  carve-out; 5e-2 heldout bar. Files: surrogate.py + surrogate_training.py +
  8 test files.
- DESIGN-TRIAGE ROUTING (INS-3-001..004, same build): doc-staleness findings
  naming BOTH SPEC.md and DATA_CONTRACTS.yaml (INS-3-001/002, stale exterior-
  polar tag) -> Librarian/doc-sync override, NOT a Coder WP (recurring rule).
  TEST-code defects -> Test-Dev scope, never Coder: INS-3-003
  (`_train_tile`/`_train_exterior_chart` hardcoded origin='low') and INS-3-004
  (`_synthetic_exterior_polar_chart` sentinel leak). CRASH-CLAIM TRIAGE
  LESSON: an Inspector finding claiming a crash at theta_hi=1.7 misread
  y=1.7 for theta_c — the fixtures' theta_c max ~0.92 rad, so there is NO
  current crash; verify the claimed value is inside the fixture's actual
  domain before routing a crash finding to a defect fix.

## Cusp ppGO fast rung build (2026-08-08/09)

- CUSP PPGO FAST RUNG DESIGN (Build cusp_ppgo_high_w): gate on control
  radius R (not w) per Simplifier — R is the correct asymptotic parameter,
  composable with envelope_bar, source-independent. Dual gate: R >=
  r_ppgo_min AND w >= w_floor (kernel-truncation guard).
  r_ppgo_min = (_R_PPGO_ERROR_CONST * _UNIFORM_ERROR_CONST / bar_ppgo)^(2/3)
  with bar_ppgo = envelope_bar/10 (calibration target). SHIPPED (as of the
  original build) with _R_PPGO_ERROR_CONST=50.0 (PROVISIONAL), _W_PPGO_FLOOR
  =50.0, _PPGO_BAR_DIVISOR=10. SUPERSEDED VALUE (2026-08-12,
  deltoid_exterior_cusp_gap WP-1): _R_PPGO_ERROR_CONST progressively
  tightened 50.0 -> 3.0 (undocumented intermediate) -> 1.0, dropping
  r_ppgo_min from ~464 to ~71.1 to ~34.2 — opens the mid-w ppGO band for
  both parities. Still PROVISIONAL; post-build driver calibration sweep
  remains owed at the current 1.0 value. _W_PPGO_FLOOR/_PPGO_BAR_DIVISOR
  unchanged.
  DO-NOTHING: fold_ppgo_correction already has internal guards;
  LensDomainError caught -> fall through to the Pearcey path. Professor
  confirms: Pearcey -> geometric image sum as (x,y)->inf; fold_ppgo_correction
  converges to the same limit; both branches (astroid+saddle) valid.

## Exterior 2D (rho, u) fold-carrier (2026-08-10)

- DESIGN (Build exterior_2d_fold_carrier): extends the 1D rho-carrier
  (b061103) to a 2D (n_rho, n_theta_c) `rho_u_carrier` on ExteriorPolarChart.
  MOTIVATION (probe): the 1D rho-only carrier left 11.66 rad phase winding
  in u (max dphase/du 48, 82 on raw theta_c); the 2D carrier flattens the
  per-rho u-phase span to <= 1.63 rad, splineable at 4 nodes/axis.
  Simplifier rulings: migrate 1D->2D at the NPZ load boundary (broadcast,
  ONE serve path); compute on theta_c_grid by index-pairing (no inverse
  interp); extract `_probe_ghost_delay` helper; NaN fill along u then rho;
  schema bump to exterior_polar_rho_u_carrier_v2; continuity gate + k_chart
  on the 2D-demodulated envelope. Professor rulings: median-over-gamma
  correct; w_grid[0] probing sufficient; tabulate (no linear fit); RAW
  absolute delays, Re(tau_c) only (NO Im(tau_c) demod — e^{+w*Im} explosive
  ~19x at w=30); tolerances node-exact 5e-13, off-grid phase 1e-3 rad,
  heldout eps 4e-3, self-falsification 10x. INS-1-002/003 (SPEC.md +
  DATA_CONTRACTS.yaml still 1D/"only known tag") = OVERRIDE -> Librarian
  doc-sync (recurring rule), NOT Coder.
- BUILD-RESUME PATTERN (this build): a build that dies at the Inspector
  stage (after coder fixes were committed but before final verification)
  is completed by a FRESH Inspector re-verify (re-run suites + re-derive
  invariants), then the Librarian post-commit sync, then the manual
  test-fix pass — same family as the quota-death salvage pattern; never
  close on the pre-crash partial pass.

## Saddle exterior full treatment (2026-08-10)

- CUSP-ADAPTED U TRANSFERS ACROSS PARITIES (Build saddle_exterior_full_treatment,
  238d21e): the deltoid cusp is the SAME A3 catastrophe as the astroid cusp
  (universal 2/3 exponent, d**(-1/3) divergence), so the cusp-adapted
  u=d**(2/3) coordinate transfers from the parity==1 astroid tiles to the
  macro-saddle (gamma>1) exterior — which previously trained raw-theta and
  failed 91/154 tiles at the 1e-3 heldout bar. RULE: before mirroring a
  coordinate treatment to the opposite parity, confirm the catastrophe class
  matches (both A3 here); if it does, the coordinate transfers — but the
  edge-anchored wedge map needs an interior-anchor generalization
  (`_deltoid_cusp_axis_map`) since the deltoid cusp sits inside the tile
  range, not on a wedge edge.
- PARITY-GATE CONSTANTS WHEN THE PHYSICS DIFFERS BY PARITY: the astroid's
  `_CUSP_ARM_COVERAGE=0.07` cusp-window shrink does NOT transfer to the
  saddle — saddle deep-interior images can sit arbitrarily close to the cusp
  (F018), so coverage ~0. Ship a parity-gated twin constant
  (`_SADDLE_CUSP_ARM_COVERAGE=0.0` placeholder) pending post-build
  calibration; never reuse the opposite parity's value. Acceptance bar
  1e-3 heldout + angular-uniformity test.
- SERVING-GEOGRAPHY RULING (Professor, same build): the deltoid straight
  edges (fold arcs) and the inter-lobe corridor need NO new serving code —
  exterior charts cover the exterior, lobe-interior charts cover lobe
  interiors, corridor falls through to the exact engine. Get the ruling
  before building.
- TREE-GATE INFRA-CRASH DEATH PATTERN (new build-death family, distinct from
  quota-death and stranded-build): this build died at the TREE gate on a
  pytest teardown INFRA crash (Pluggy 'cannot send (already closed?)' after
  the -n2 retry timeout) — NOT a code failure; Inspector PASS + Professor
  PASS; salvaged manually. An infra-crash death AFTER full agent PASSes
  closes on manual salvage without a re-audit.

## 2026-08-11 builds (ppGO resolution gate, Pearcey residual table, mpmath GL)

- PPGO RESOLUTION GATE DESIGN (Build operator_routing_one_home): ppGO rung
  in cusp_amplification gains a dual gate — serve via fold_ppgo_correction
  iff (_merging_fold_pair(...) is not None) OR (w*delta_min >=
  _PPGO_RESOLUTION_GATE = 4.0, mirroring operator.RHO_END). Fold-pair nodes
  (Morse 0,1) serve regardless of resolution; saddle-only nodes (Morse 2,3)
  need the resolution gate; on gate miss result=None -> falls through to the
  Pearcey uniform form. ONE Coder WP; Professor + Simplifier lean. Professor
  confirmed the spec's w*delta_min~1.9 estimate was a copy-paste error
  (fixture gives 322 at w=500; saddle sources always resolve at w>=50).
- PEARCEY RESIDUAL TABULATION (Build zero_quadrature_pearcey): R(x,y) =
  P(x,y) - P_asymp(x,y) replaces demodulated tabulation; schema bump 0.2.0
  hard-refuses old artifacts; demodulate/remodulate/_carrier_phase/
  _dominant_stationary_point deleted. WATCH: P - P_asymp is DISCONTINUOUS
  across a caustic crossing — a spline residual cert sweeping the caustic
  blew to 1.9e+09; residual tabulation is topology-region-local.
- INTERIOR CUSP SERVING BARRIER (Build interior_cusp_serving_barrier): skip
  _calibration_certified for interior (3 stationary points) — uniform error
  gate.
- _CUSP_VERTEX ROUTING FIX (Build revert_residual_table_fix_routing): probe
  all nearby cusps by source-plane distance; ONE Coder WP, no NPZ regen.
  Side effect: _cusp_vertex now returns a finite wedge-tip vertex where old
  code returned None at wedge-edge configs -> 8 pre-existing vertex tests
  red at HEAD (separate committed build, not this one's diff).
- FIXED-PANEL MPMATH RULE (Build mpmath_fixed_panel_rule): replace mp.quad
  with fixed-order composite GL (mp.gauss(24), no Newton fallback —
  Simplifier trim) in _raw_integral_mp; lru_cache on (order, dps) at module
  level; N/2N certification stays on RECONSTRUCTED F (raw I underflows at
  w=150). Professor: order-24 sufficient (12 nodes/wavelength; mpmath dps
  >> dd); spot grid w∈{61,80,100,120,150} x gamma'∈{0.3,0.7,1.5} x
  y_eig∈{(0.1,0.1),(0.4,0.3),(0.8,0.5)} = 45 pts; tol = _CERTIFICATION_TOL
  = 3e-10 on reconstructed F; complex(raw_n) safe; O(seconds) deterministic.
  Optional: gamma'=1.05 edge case at w=80 for near-parity boundary stress.

## 2026-08-12 builds (lobe_exterior region wiring, deltoid exterior geometry, cusp gap)

- LOBE_EXTERIOR REGION WIRING (Build lobe_exterior_region_wiring, Option A —
  COMPLETE): `lobe_exterior` shipped as a NEW first-class training region
  alongside tube/exterior/wedge_interior/lobe_interior. ONE Coder WP, 5
  edits: `_self_estimate` default region tuple + per-region cost dict
  (cost=1, same as lobe_interior); `_train_band_charts` default region
  tuple; saddle/lobe admissions gate widened from {'exterior'} to
  {'lobe_interior','lobe_exterior'}; packing-loop tile-block gate widened
  'exterior'->'lobe_exterior'; positive-parity exterior block left BYTE-
  IDENTICAL. CLI: scripts/train_lens_surrogate.py --regions choices
  extended. NPZ tag infix is `_fflobeext_`, decoded FIRST in `_tag_kind`
  (before `_ff_`, since neither is a substring of the other — avoids
  substring-leak false positives). Test Dev: 2 disjoint shards (A = region-
  filter tag/estimate tests incl. rewriting test_exterior_only_saddle to
  EMPTY since exterior far-field is parity==1-only and adding
  test_lobe_exterior_only_saddle; B = slow-operation-guard pin-tuple
  extension). Foreman-Lite fixed one stale doc-comment (INS-7-001: the
  LobeExteriorChart `theta_to_u` load is a SOFT read via `data.get()`,
  mirroring LobeInteriorChart's convention — NOT a hard `data[...]` read
  like the wedge loader; earlier in-flight comment wrongly called it "wedge
  convention... read hard"). Inspector PASS (3 passes); Professor PASS
  (54 tests, 13.7s). Carry-forward to Librarian (non-Coder, doc-sync):
  SPEC.md + DATA_CONTRACTS.yaml have ZERO mention of the lobe_exterior/
  lobe_interior/wedge_interior region vocabulary despite lobe_exterior now
  being a public --regions CLI choice + NPZ kind + training-region contract.
- DELTOID EXTERIOR GEOMETRY FIX (Build deltoid_exterior_geometry_fix,
  re-planned 2026-08-12 — an earlier plan draft was explicitly superseded,
  discard any prior note referencing corridor_half exclusion or origin-
  centered corridor coords): corrected design charts the deltoid exterior
  in LOBE-LOCAL coordinates `(rho_lobe, u=d**(2/3))`, reusing
  `_lobe_cusp_axis_map` rather than inventing a new map. The inter-lobe
  corridor is served directly by the +y1 lobe's exterior chart — no
  separate corridor_half exclusion band needed. Cusp exclusion reuses the
  existing `caustic_cloud` nearest-distance test with
  `eta_max = f_max * R_c` (same currency as the tube-shell admission gate,
  not a bespoke frame). Admission is image_count==2, KERNEL_SUM label only
  (no ghost term at this stage — that's the follow-on cusp-gap build).
  Oracle = Schwinger engine. New `LobeExteriorChart` dataclass carries
  `theta_to_u` from construction (not retrofitted); `_SaddleLobeAdmission`
  gains an `admits_exterior` method — 2-ARG signature `(center, half)`,
  DISTINCT from `_InteriorAdmission.admits_exterior`'s 3-arg
  `(center, half, source_magnitude_max)` — never conflate the two when
  reusing/extending admission logic. WP1 surrogate.py (chart class), WP2
  surrogate_training.py (tiler/packing), WP3 census (routed to
  Foreman-Lite, not Coder — census-only fallthrough-category work).
- DELTOID EXTERIOR CUSP GAP — OPTION C HYBRID (Build
  deltoid_exterior_cusp_gap, planned + executed 2026-08-12): Professor's
  recommended hybrid closes the near-cusp exterior coverage gap left by the
  geometry-fix build. WP-1: lower `r_ppgo_min` by tightening
  `_R_PPGO_ERROR_CONST` (50->3->1.0, see the ppGO fast-rung section above
  for the superseded-value history) — closes the astroid mid-w exact-
  flashback gap and serves saddle sources outside the fold band; one-
  constant change in `_pearcey_cusp.py`, calibration sweep owed post-build.
  WP-2: extends the surrogate to the saddle exterior cusp window via a new
  MINUS_GHOST label (`force_minus_ghost` flag threaded through
  `_build_farfield_chart` -> `build_ff` closure -> `_subdivide_farfield_tile`
  .build_child -> `_subdivide_tile` child dict) with a REDUCED
  `_CUSP_EXCLUSION_DISTANCE` for near-cusp saddle-parity tiles (set to 0.0
  at the tiler; the per-node ghost-gate + eps bar decide admission instead
  of a blanket radius exclusion — "let label gates + eps bar decide", not
  YAGNI since MINUS_GHOST=0.35 is a real KERNEL_SUM-invalidating measurement
  near cusps). No serve-side changes needed (MINUS_GHOST already handled at
  serve). Saddle sources strictly inside the fold band (nearest.distance <
  0.3) stay excluded from the ppGO rung even after WP-1 — WP-2 is required
  for those; WP-1 alone closes most of the gap for BOTH parities. WPs are
  independent (different files/layers), WP-1 sequenced first; no second
  ppGO rung needed (Simplifier).
  TRIAGE INS-1-001 (coder_fix, 2026-08-12): WP-2's `force_minus_ghost`
  training crashed with a per-node `GhostDomainError` because
  `farfield_envelope_from_partition` (the function that implements
  MINUS_GHOST) was called OUTSIDE the node-level `except _REFUSAL_ERRORS`
  guard in `from_engine` — the tile-level far-field except clause only
  caught `CarrierDiscontinuityError`. Fix, two parts: (1) PRIMARY — move
  the ghost-gate call inside the node-level try/except so a refusal becomes
  a per-point refused-node (preserves partial-tile coverage, matching the
  "let label gates decide" design intent); (2) SAFETY NET — also add
  `GhostDomainError` to the tile-level except tuple alongside
  `CarrierDiscontinuityError`, routing a whole-tile miss to the
  ladder-served gap — NEVER route a ghost-gate refusal into
  `_subdivide_farfield_tile`, since the refusal is a GEOMETRIC boundary
  (near-cusp), not a resolution problem that subdivision could cure.

## 2026-08-13 builds (ppgo_interior_certificate, fold_exterior_ghost)

- PPGO_INTERIOR_CERTIFICATE: re-gated the interior fold-ppGO rung in
  `likelihood.py` onto TWO legs — (1) exact interior predicate
  `int(geom.real_mask.sum()) == 4` for both parities, replacing the rho<=1
  leg AND the saddle-only `!=4` guard; (2) a new
  `geometry.ppgo_error_estimate(real_images, source, matrix, w_min)` =
  `sum_a sqrt|mu_a| * |c3_a| / w_min**3`, admitted at
  `est * _PPGO_INTERIOR_SAFETY(=2.0) <= CERTIFICATION_BAR`. The xi/fold-pair
  leg was DROPPED BY MEASUREMENT (all 78 interior configs failed it while
  the certificate admitted 230 band rows at max true err 4.8e-5). Design
  lesson: when a rung's own asymptotic series is available, the certificate
  is a series term — not a geometric proxy. On true interior the ghost is
  exactly zero (GhostAbsentError), so the rung carries NO ghost term.
- FOLD_EXTERIOR_GHOST: two-sided. (A) fold refusal `len(images) != 4` placed
  at the three ENTRY POINTS (`fold_amplification`, `fold_ppgo_correction`,
  `channels.born_carrier_from_partition`), NEVER inside the shared
  `_merging_fold_pair` primitive — it has 5 consumers and tightening it
  would flip an unrelated `_pearcey_cusp` disjunct. GUARD PLACEMENT RULE:
  put a new refusal at the consumers you intend to change, not in the shared
  primitive they happen to share. (B) new `_ghost_ppgo_amplification` rung
  in `_uniform_arm_value`, ordered fold -> ghost+ppGO -> cusp (interior fold
  first, cusp catch-all last). Measured acceptance (45 oracle points): gates
  partition cleanly, max admitted rel_err 1.98e-6 against a 1e-2 arm bar,
  zero overshoot, so no w-floor was needed. The DECAY gate was the sole
  active discriminator on that grid; admit/refuse tracked Im(tau_c), NOT the
  |y|/rc label — never describe such a gate by the label band it correlates
  with.
- DEFERRED-TRAINING MIRRORS ARE PART OF THE SAME FIX: `surrogate_census`
  mirrored the RETIRED xi gate for a whole build after `likelihood.py` moved
  to the c3 certificate. Plan the mirror re-gate inside the build that
  changes the production rung, or it lands a build late as a laggard finding.

## 2026-08-14 builds (symmetry_tie_c3_admission)

- SYMMETRY_TIE_C3_ADMISSION: the saddle far-field serve gate was re-keyed a
  SECOND time in one day — first onto a directional eta floor (measured-
  boundary rule), then that plan was fully reverted in favor of a direct
  c3-certificate (S*ppgo_error_estimate(w_lo)<=bar, S=20, bar=1e-3) + image-
  separation backstop (0.05), retiring `_SADDLE_FARFIELD_RHO_FLOOR` and the
  eta floor entirely. Confirms the 2026-08-13 ruling "a gate must bound the
  error of the object it admits" over a geometric proxy (eta) — direct error
  estimation from the served object's own asymptotic series won out over a
  second geometric-distance attempt. Both call sites + all saddle/ppgo test
  files + the census mirror were re-keyed in the SAME build (mirror-
  currency-with-production rule holds).

## 2026-08-14 build (tube_d2_fold)

- TUBE D2-FOLD: reflection is parity-agnostic (same astroid+saddle formula;
  saddle interpretation = lobe-swap 0<->pi for s1, branch-swap for s2).
  CORRECTION to an earlier same-day claim: fold is NOT bit-exact across all
  4 sign octants uniformly — only the negation-only octant pairs are
  bit-exact (IEEE-754 sign flip is exact); the pi-theta reflection octant
  pairs differ by ~1 ULP (the `pi - theta` subtraction rounds). Fundamental-
  domain query (s1=s2=+) vs unfolded incumbent stays bit-exact (fold is
  identity there); folded-arc vs the OLD reflected-arc serve is rtol 1e-6
  only (independently integrated theta_to_s) — transitional test, not a
  durable pin.
- Professor CORRECTED handoff fact: select_chart runs tube FIRST, so saddle
  tube is live+unfolded (6 arcs reachable) — the half-ring hole (F079)
  exists on saddle too, not just astroid. Decision: SERVE-fold both
  parities (closes F079 both, enables equality pin both); TRAINING reduces
  astroid arcs 4->1 only (fundamental arc = caustic_theta in (pi/4,3pi/4),
  the arc bracketing pi/2); saddle training stays at 6 arcs (F079 closes
  via serve fold alone).
- Simplifier: fold implemented tube-local via the production functions the
  census already reuses -> census stays auto-current, no census code
  change needed. _EXPECTED_ARCS {1:4,-1:6} left UNCHANGED (topology guard
  in detect_caustic_structure is a separate gate from the training-slice
  reduction).
- ONE Coder WP owned both surrogate.py serve fold and surrogate_training.py
  fundamental-arc selection — single head for "which arc is fundamental"
  convention avoids a two-owner split-brain.
- TRIAGE INS-1-001 (coder_fix): classify_fallthrough's tube branch had TWO
  `_tube_serves` call sites; only one (the exterior-polar sibling) had been
  threaded with the new y1_eig/y2_eig fold args — an internal miss, not a
  design ambiguity. Generalizes: when a design adds args to a shared call,
  grep ALL call sites of that function name, not just the one the brief
  names.

## 2026-08-14/15 build (tiling_census_node_budget)

- ENGINE-FREE census design pattern: reuse existing tiler fns as THIN
  CALLERS (never reimplement selection/counting logic) — parallel
  reimplementation is the F-class defect the census exists to avoid.
  When a census necessarily omits a downstream trim step (e.g. per-node
  ppGO trim) because mirroring it would require re-implementing DROP
  DECISION logic (a second copy of decision logic), the right triage is
  DISCLOSE the one-directional (conservative, over-count only) divergence
  via docstring + an explicit output field (e.g. `ppgo_trim_modeled:
  False`) rather than fully mirroring it — full fidelity isn't worth the
  duplication-drift risk for a pre-campaign gate.

## 2026-08-14/15 build (saddle_tube_fundamental_training, F081)

- v1 plan REJECTED: it enshrined the band-wide max(arc_r_min) as an
  invariant to PRESERVE — but that max IS the F081 starvation defect
  (an outlier outer-arc r_min balloons the shared admission shell and
  starves lobe/far-field tiles). Lesson: when a fix's own acceptance
  criterion is "preserve invariant X" and X is exactly the quantity the
  triggering defect report says is wrong, re-derive the acceptance
  criterion from the defect report, don't just port the old value
  forward.
- v2 (shipped): Part A = orbit-partition trim of per-parity training arcs
  by D2 midpoint-angle clustering (retiring the max_tube_arcs knob
  entirely — count follows the partition, never hardcoded); Part B =
  route the two SHARED admission scalars (saddle lobe eta_max, far-field
  exclusion_rho) to the REGION-ADJACENT min(arc_r_min) instead of the
  band-wide max(arc_r_min); tube w-cap + astroid interior-skip/wedge
  extent correctly KEPT on the max (different consumer, different
  currency).
- Measured outcome (saddle band, real geometry): 6 detected deltoid arcs
  -> 2 D2-orbit representative arcs (NOT the naive a-priori guess of 3;
  orbit sizes {4,2}). Confirms the standing "DERIVE, don't hardcode"
  rule — verify via independent union-find, not by eye. arc_r_min
  anisotropy ~23x (0.399 to 9.156) on this fixture.
- Cross-build consumer-suite handling: Coder correctly deferred 4 broken
  test files (signature/field removal) to Test Dev AS FLAGGED findings;
  Test Dev picked them up same session; Inspector's pass-2 required
  RE-DERIVING each expectation from the live production selector (not
  just deleting/loosening the old assertion) before granting PASS.


## 2026-08-15 build (lobe_cusp_axis_edge_tolerance, F082 smoke crash)

- 7a smoke crash in `_lobe_cusp_axis_map` (surrogate.py): `_lobe_nearest_cusp`
  picks `side` from tile CENTER while the cusp-vs-edge guard was STRICT
  (raises on machine-precision straddle, cusp 3.27e-16 vs theta_hi 3.55e-16,
  2.8e-17 inside). Shipped fix: option (a), tolerance-relax the edge guards
  (`_CUSP_EDGE_COINCIDENCE_ULPS = 8`) + clamp d->0 at coincidence, keeping
  the function's non-Optional return type — NOT option (b) mirroring
  `_deltoid_cusp_axis_map`'s Optional-straddle->None pattern, because 3
  production callers don't None-handle and `_chart_from_npz` has a known
  latent unconditional `data['theta_to_u']` KeyError trap that a None
  return would walk straight into. Sibling audit confirmed only
  `_lobe_cusp_axis_map` had this defect shape — `_wedge_cusp_axis_map`
  pins cusp to the domain edge by construction (no guard needed) and
  `_deltoid_cusp_axis_map` already handled coincidence gracefully.
  Professor + Inspector both PASSed the fix; Librarian confirmed no
  SPEC/DATA_CONTRACTS impact (private-helper guard tolerance, not a public
  contract change).

## 2026-08-15/17 build (serve_route_census)
- ENGINE-FREE CENSUS MODULE PATTERN: model new engine-free demand/census
  modules on `tiling_census.py` (lazy imports), never `surrogate_census.py`
  (imports engine at module load). MECE waterfall taxonomy: decision ORDER
  can differ from the published label order (put engine_refused first even
  if it's not first in the label list) — document both explicitly.
- RESIDUAL/ROUTE SPLITS MUST GAUGE ON caustic_rho, NEVER rho_lobe (F073
  lineage) — this recurs across builds as the correct partition currency
  for anything measuring distance-to-caustic outside the astroid interior.
- ROUTE-EQUALITY PINS (e.g. D2 sign-flip invariance) belong on the route
  KIND vector, not a lobe/branch index — index identity is parity-fragile,
  kind identity is the physically invariant object.
- NEVER HARDCODE AN EMPIRICAL RATE CLAIM (e.g. "engine_refused ~59%") into
  a plan or docstring as a predicted constant — require it be MEASURED and
  reported per-run; sample-dependent rates are report output, not a design
  invariant (converges with Professor's "must be reported empirically,
  never asserted a priori" ruling).

## 2026-08-17 build (tube_beat_free_representation, multi-launch recovery)
- ERROR-METRIC CURRENCY (generalizes GATE CURRENCY to error metrics, not
  just admission gates): an accuracy sweep on an object that is itself an
  interpolated RESIDUAL (e.g. r=E/F_ref) must normalize by the reference
  the interpolant controls (F_ref), never by the raw physical total — the
  raw total can vanish at points unrelated to interpolation error (e.g. an
  old carrier's Airy zeros), producing a false failure signature.
