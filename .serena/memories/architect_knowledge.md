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
- Doc-sync findings are never Coder WPs: route to post-gate Librarian with
  exact replacement text; SPEC.md in files_affected is informational.
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
  delay/index helpers; gate activation on a physically meaningful currency
  (e.g. w·Im tau_c) mirroring existing threshold-constant conventions, plus
  explicit degenerate-axis refusal (Build 8h-b).
- Surrogate/emulator design (Build 8a): emulate the SMOOTH symmetry-
  invariant object (the beat-free envelope E(w)), NOT the oscillatory total;
  build ONE interpolant PER topology region (parity/image-count) since the
  decomposition changes topology at caustics; exact-engine fallback near
  caustics + outside the box. Reduce out any EXACTLY symmetry-eliminable
  parameter (beta via eigenframe rotation) BEFORE training — lower the
  surrogate dimension, don't train over it.
- Conservative-serve gate = axis-aligned box containment + exclusion balls
  around refused points + per-sample refusal propagation; NEVER a learned
  mask (a false negative is a correctness bug, not an efficiency miss).
  Default surrogate=None -> exact path byte-identical; enable-by-default
  deferred pending full-box artifact + census + PP-plot.
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
  from an accuracy floor before proposing to raise a ceiling constant —
  raising the wrong one buys nothing and can cross into the wrong regime.
- A finding whose fix is confined to test-file fixtures/constants routes to
  Test Developer, never Coder — recurring precedent (4x+).
- Feed Inspector-authored fix snippets through Simplifier before endorsing
  verbatim — a shape mismatch (e.g. dict vs flat-list) can ship a fix that
  passes in isolation but breaks the existing consumer contract.
- Grid/node reprovisioning: reuse an existing normalized held-out-error
  metric (e.g. LOO) to decide how many nodes to keep/drop rather than
  hardcoding a reduction heuristic (e.g. a flat 2x) — let a probe decide
  (Build 8h-b).
- When an accuracy/interior label becomes ill-conditioned in a parameter
  sub-region (e.g. near a higher-order catastrophe), plan a switch to an
  alternate ALREADY-ESTABLISHED label/envelope for that sub-region with a
  concrete falsifiable pass/fail pair, rather than tuning the ill-
  conditioned label further (Build 8h-b).
- New accessors added to an existing family (e.g. w_ceiling alongside
  w_cert/w_trust) should mirror that family's naming/behavior exactly
  rather than invent a new sentinel type (Build 8h-b).
- A WP framed as "find a bug that a test might expose" with no pre-
  identifiable defect is not a valid Coder WP — it's a forbidden measure-
  and-decide campaign; a repair already committed but unexercised by
  tests is a Test-Dev completion/port task, not Coder (Build 8h-b5).
- When a WP is explicitly redirected off the default agent for a routing-
  precedent reason (e.g. Coder -> Test Developer), name the executing
  agent directly in coder_instructions — an implicit redirect risks
  silent mis-route back to the default agent (Build 8h-b4).
- When multiple pipeline stages (e.g. tiler/chart/serve) each need the
  same physical quantity, require them to call one identical shared
  function — independent reimplementations invite serve-mirror
  divergence (Build 8h-b3-FIN).
- When two sibling code paths (e.g. interior/exterior admission) derive
  from the same shared geometric anchor (e.g. caustic cusp rays),
  symmetrize a structural fix (e.g. cusp-alignment) across BOTH rather
  than patching one — an asymmetric fix inherits the same kink the other
  side already solved (Build 8h-b6).
- A gate relaxation that is only sound GIVEN a prerequisite structural fix
  (e.g. a coarse multi-probe test collapsing to one representative probe
  is only valid once directions are pre-aligned) must be planned as ONE
  merged WP with that prerequisite — landing the relaxation alone is
  unsound (Build 8h-b6).
