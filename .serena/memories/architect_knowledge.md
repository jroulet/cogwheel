# Architect Long-Term Knowledge

- When the source task/prompt is unreadable in-session (sandbox confines
  Serena file tools to the project root, built-in Read/Bash disabled,
  `execute_shell_command` absent, orchestrator log gitignored/blocked),
  do NOT fabricate a task from guesswork. Emit a plan with empty
  `work_packages` and a summary flagging the blocker for a human re-run.
  Confirm via more than one independent attempt first.
- Before planning a docs-only task, check whether the target already
  substantially satisfies the goal. Plan the minimal diff, not a
  rewrite-from-scratch.
- Simplifier verdict pattern: prefer the leanest correct response over
  speculative extra work — no no-op verification WPs "just in case".
- Batching an evaluation grid where all non-grid params are fixed: hoist
  grid-independent quantities once and consider algebraic regrouping
  (e.g. bilinear form -> precomputed per-order weight vectors dotted per
  node). Any accumulation-order change needs re-certification vs an
  independent oracle plus a certify-vs-refuse decision-identity check
  (solo vs batch).
- Single certification path: make the scalar API a thin wrapper over the
  batched core, so one path carries all refusal logic and existing
  scalar-tolerance suites automatically exercise the new code.
- Timing acceptance: lead with machine-independent structural gates
  (component subdominance, speedup floor); an absolute ms ceiling should
  be arithmetic-derived (measured floor x margin), never
  machine-calibrated — the owner dislikes machine-cal ceilings.
- When a perf target is unreachable without a deferred design (e.g. a
  surrogate table), document the measured floor honestly and escalate to
  the owner; never widen tolerances or fake a ceiling — prefer a
  slower-but-correct default.
- Known FINDINGS bug patterns recur in sibling code paths (e.g.
  real-only vs full-cluster candidate sets, F008): when diagnosing slow
  convergence or misplaced nodes, grep for the same pattern elsewhere
  before inventing a new mechanism.
