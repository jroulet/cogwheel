# Test Dev Long-Term Knowledge

- Premise repair, not tolerance repair: if a fixture assumes a wrong
  physical premise (e.g. F->1 at w->0 under shear), fix the fixture to
  a case where the premise holds (gamma=kappa=0); never widen tolerance.
- Anti-dodge pairing: keep the inconvenient original case as a
  companion test that PREDICTS its nonzero offset via an independent
  closed form.
- To inject a buggy/old variant without editing source,
  mock.patch.object the MODULE GLOBAL the function resolves internally.
- Extend AST/name-forbidding guards for every new mutation/oracle
  helper (oracle independence, F002).
- For rules differing only in edge cases, assert a sub-case where old
  and new logic must agree bit-for-bit as a cheap regression control.
- Fully revert probe/mutation edits; verify by read-back plus a
  pattern search for residue.
- Shell gate is command-shape sensitive: plain `python -m pytest <file>
  -q` may pass while heredocs/`python -c`/pipes are denied — prefer the
  plainest working shape.
- A bare unexplained denial is often transient — retry once; a denial
  WITH an explicit reason binds and must be respected.
- In a linked-worktree repo, run test commands from the WORKTREE root.
- Falsification under numba: patch through the FULL .py_func chain
  (F010); assert .py_func bodies lack .signatures so removing @njit
  can't make the test vacuous; define "gate RED" as (refusal raised) OR
  (error > tol) — a perturbation may surface as a refusal, not a wrong
  value.
- Test refusals at the production default operating point (e.g. default
  max_order), not at the accuracy-study setting where they converge.
- When a plan-anticipated gate exposes a production shortfall, leave it
  RED (no tolerance widening, no production edits); repoint the
  positive control to a converged configuration so the falsification
  stays non-vacuous and green.
- Different pipeline paths have different numerical floors (e.g. RB
  binning vs brute force) — gate each path at its own floor.
- An aggregate downstream gate can pass while a component gate fails
  (error budget dominated elsewhere) — keep both; the component gate
  protects regimes the fixture doesn't reach.
- Prefer machine-independent structural timing gates (speedup ratio,
  component subdominance); absolute ms ceilings only as
  arithmetic-derived secondary regression guards.
