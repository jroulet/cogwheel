---
section: Backlog
---
- **Declare and modernize the Claude Agent SDK dependency** `[housekeeping]` — the
  SDK version is not declared anywhere: no pin in `pyproject.toml`, no
  requirements file, nothing. `0.1.53` is what is installed in the active SDK
  environment as verified 2026-07-24. An undeclared dependency is worse than
  an outdated pin: a routine upgrade silently changes the pipeline's behaviour,
  and a fresh install gets whatever PyPI serves that day.
  **Why this is load-bearing, not hygiene.** Three orchestrator behaviours exist
  ONLY as workarounds for the 0.1.x anyio semantics, and all three cost real
  throughput: (1) `_run_dag` runs DAG nodes sequentially, (2) `_run_coders` runs
  each batch's work packages sequentially, and (3) `_iter_query_with_timeout`
  drains the stream through a queue in one dedicated task. All three exist
  because `query()` holds anyio cancel scopes across yields, so concurrent
  streams — or even one stream resumed from a fresh task per message — blow up
  with `RuntimeError("Attempted to exit cancel scope in a different task")`. The
  pipeline is therefore fully serialized: a six-WP build runs six coders one
  after another when the dependency graph permits parallelism.
  **Known obstacle.** A blind `pip install -U` to `0.2.119` was tried on
  2026-07-16 and the pipeline died with `Control request timeout: initialize` —
  the 0.2.x line changed the control-protocol handshake. So this is a real
  migration, not a version bump, and that is why it is a fragment rather than a
  drive-by.
  **Shape of the work.** (a) Declare the dependency explicitly (a test/dev extra
  in `pyproject.toml`, or a requirements file the launcher checks) so the version
  is a decision rather than an accident. (b) Move to a current 0.2.x, fixing the
  initialize handshake. (c) Re-test whether the cancel-scope hazard still exists
  there — if the SDK no longer holds scopes across yields, DELETE the three
  workarounds and restore intra-batch parallelism rather than carrying them
  forever; the comments on each say not to revert without re-testing against the
  SDK in use, and this is that re-test. (d) `.claude/sdk/tests/test_iter_query_stream.py`
  pins the drain contract and should keep passing or be retired deliberately.
  **Sequencing.** Do this AFTER the lensing builds prove green on the current
  stack — changing the SDK underneath an unproven pipeline would confound the
  two. Once cogwheel proves it, port to `~/Work/teja-force` (assets/sdk) and gw.
