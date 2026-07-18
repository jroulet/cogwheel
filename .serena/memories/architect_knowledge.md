# Architect Long-Term Knowledge

- If the source task/prompt is unreadable in-session, do NOT fabricate a
  task — emit an empty-WP plan flagging the blocker after >1 independent
  read attempt.
- Docs-only tasks: check whether the target already satisfies the goal;
  plan the minimal diff, not a rewrite.
- Simplifier verdict pattern: leanest correct response; no no-op
  verification WPs "just in case".
- Verify any "code-pinned"/"already exists" claim with fresh find_symbol/
  grep BEFORE planning WPs on it (fabricated pin stalled Build 3e); agents
  refusing to fabricate a missing primitive = plan failure, not agent failure.
- Doc-sync findings are never Coder WPs: route to post-gate Librarian with
  exact replacement text attached; SPEC.md in files_affected is informational.
- Don't escalate a perf/accuracy floor measured on a defective build —
  fix, retune, re-measure, then escalate.
- Interpolation node budgets scale with oscillation content (cycles in
  band), not kink count; have the Professor sanity-check brief perf
  arithmetic; ship the cheap structural lever before a surrogate table.
- Batching: hoist grid-independent quantities; any accumulation-order change
  needs re-certification vs an independent oracle + solo-vs-batch
  certify-XOR-refuse identity. Scalar API = thin wrapper over batched core
  (single certification path).
- Timing acceptance: machine-independent structural gates first; absolute
  ms ceilings must be arithmetic-derived, never machine-calibrated.
- Unreachable perf target: document the measured floor honestly and
  escalate; never widen tolerances — slower-but-correct.
- Known FINDINGS bug patterns recur in sibling code paths — grep before
  inventing a new mechanism.
- Refusal boundary: thin Posterior-subclass override maps named domain/
  cancellation refusals -> -inf + metadata; raw likelihood keeps its raise
  contract; the sampler never catches (Build 4).
- Posterior requires prior.standard_params == likelihood.params EXACTLY —
  plan the prior layer against the likelihood's params; fix unmeasurable
  params via FixedPrior rather than omitting them.
- Cache determinism: snap proposals to module-constant lattices so a
  fiducial/cache entry is a pure function of the candidate (memoize on the
  lattice key).
- Prefer one-line guards + fallback-to-certified-direct over topology-aware
  partitioning (Simplifier trim, Build 3g).
- Derived in-memory caches: plan a __getstate__ that drops them (bases
  define none); JSONMixin/get_init_dict path is unaffected.
- Reuse an existing prior with a documented option-deferral rather than
  block a build on the ideal coordinate (d_app deferred to Build 5).
