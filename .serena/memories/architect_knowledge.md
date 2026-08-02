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
  (here "far-field" = a trained chart GAUGE, NOT weak deflection) — pick a
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
- Arc-length reparametrization design: use rep_gamma = median(gamma_grid)
  to minimize worst-case effective excursion; a single-gamma map is adequate
  for topology-stable bands (degrades only near parity wall where other
  guards already bound the regime).
