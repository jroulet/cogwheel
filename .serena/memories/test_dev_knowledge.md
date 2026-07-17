# Test Dev Long-Term Knowledge

- Premise repair, not tolerance repair: if a test's fixture assumes an
  incorrect physical premise (e.g. F->1 at w->0 for ANY shear/
  convergence), fix the fixture/candidate to isolate a case where the
  premise is actually true (e.g. gamma=kappa=0), not widen the tolerance
  to paper over a real, nonzero physical offset.
- Anti-dodge pairing: when repairing a premise, keep the inconvenient
  original case too, as a companion test that PREDICTS its nonzero
  offset via an independent closed form — this guards against the two
  regressions the repair would otherwise invite (normalizing the real
  effect out, or reintroducing the wrong short-circuit).
- To falsification-test a buggy-vs-fixed code path without editing
  source, `mock.patch.object` the MODULE GLOBAL a function looks up
  internally (e.g. `evaluate` resolving a helper as a module attribute)
  — this cleanly injects an old/buggy variant for the test.
- Extend AST/name-forbidding guards whenever a new mutation-test
  reproduction helper is added, so an automated check pins that it never
  references the module-under-test's own names (oracle independence).
- For a rule that only differs in edge cases (e.g. a restricted vs. full
  candidate set), find and assert a sub-case where old and new logic
  must agree bit-for-bit — a cheap, strong regression control alongside
  the falsification test.
- Fully revert any probe/mutation edit used only to surface a hidden
  numeric value in an assertion message, and verify via read-back plus a
  pattern search for residue.
- This environment's shell/tool-execution gate can be command-shape
  sensitive: a bare `python -m pytest <file> -q` may succeed when
  heredocs, `python -c`, or piped/filtered variants (`-k`, `| grep`,
  `| tail`) are denied. Prefer the plainest working shape; don't burn
  retries on inline-code probes.
- A bare "user doesn't want to take this action" denial with no reason
  is often a transient artifact — retry once. A denial WITH an explicit
  reason (e.g. "use Serena for shell") binds and must be respected, not
  retried.
- In a linked-worktree repo, run test commands from the WORKTREE root,
  not the original repo path — the latter fails with file-not-found even
  though it looks like the same project.
