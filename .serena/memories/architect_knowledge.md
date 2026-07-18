# Architect Long-Term Knowledge

- If the source task/prompt is unreadable in-session (sandbox blocks every
  read path), do NOT fabricate a task — emit an empty-WP plan flagging the
  blocker, after confirming via more than one independent attempt.
- Before planning a docs-only task, check whether the target already
  substantially satisfies the goal. Plan the minimal diff, not a rewrite.
- Simplifier verdict pattern: prefer the leanest correct response over
  speculative extra work — no no-op verification WPs "just in case".
- Verify any "code-pinned" / "already exists" claim in a brief or plan with
  fresh find_symbol/grep BEFORE planning WPs on it; a fabricated pin stalls
  the whole build (Build 3e). Coders/Test Devs refusing to fabricate the
  missing primitive is correct behavior — treat it as plan failure, not
  agent failure.
- Doc-sync findings (stale SPEC/FINDINGS narrative after a mechanism swap)
  are never Coder WPs: route to the post-gate doc-sync/Librarian phase and
  attach the exact replacement text in the triage so content survives.
  SPEC.md in a plan's files_affected is informational, not a WP instruction.
- Don't escalate a perf/accuracy floor measured on a build with a known
  implementation defect — fix, retune, re-measure, then escalate.
- Interpolation node budgets are set by the oscillation content of the
  interpolated object (cycles in band), not by kink/landmark count — have
  the Professor sanity-check brief perf arithmetic; ship the cheap
  structural lever alone before reaching for a surrogate table (owner
  directive).
- Batching an evaluation grid: hoist grid-independent quantities once and
  consider algebraic regrouping. Any accumulation-order change needs
  re-certification vs an independent oracle plus a certify-vs-refuse
  decision-identity check (solo vs batch).
- Single certification path: make the scalar API a thin wrapper over the
  batched core, so one path carries all refusal logic.
- Timing acceptance: lead with machine-independent structural gates
  (component subdominance, speedup floor); absolute ms ceilings must be
  arithmetic-derived, never machine-calibrated (owner dislikes those).
- When a perf target is unreachable without a deferred design, document the
  measured floor honestly and escalate; never widen tolerances or fake a
  ceiling — prefer slower-but-correct.
- Known FINDINGS bug patterns recur in sibling code paths (e.g. real-only
  vs full-cluster candidate sets, F008): grep for the same pattern
  elsewhere before inventing a new mechanism.
