---
date: 2026-07-28
section: Backlog
---

- **The Tidy role has been silently failing since 2026-07-19** — RESOLVED,
  with the original diagnosis corrected on two points.

  **What the TODO got wrong.** (1) `tidy_short_term.md` was NOT still 87 bytes;
  by 2026-07-27 it held a substantial entry (house line limit is 79 and
  universal; two-space sentence separators to preserve when rewrapping; a safe
  rewrap recipe; the note that two section banners are unwrappable without a
  content change). (2) The role was not "failing" at all — it ran on 2026-07-27
  and produced real work, the 153-line reflow of `cogwheel/lensing/**` that
  landed inside `66a0100`.

  **Actual root cause: a fossil, plus a missing reminder.**
  `.claude/agent_state/tidy.json` read `status: "failed", last_run:
  2026-07-19` because that was the last time the IN-DAG tidier ran — the run
  whose `error_max_turns` finalization raised an anyio cancel-scope
  `RuntimeError` in a different task and tore down the whole build DAG
  (observed twice, 2026-07-18, Build 6 attempts 5-6). The response was to make
  the in-DAG run opt-in (`SDK_RUN_TIDIER=1`, default OFF) and move style to
  post-commit advisory mode. The orchestrator's skip branch then returned
  early WITHOUT touching state, so the failure status from the crashed run
  froze in place and every driver reading it concluded the role was broken.
  The causality is the reverse of the obvious reading: builds did not crash
  and prevent the Tidier from running; the Tidier ran and crashed the builds.

  **Fixes.**
  - `.claude/sdk/orchestrator.py`: the skip branch now records
    `status: "skipped_in_dag"` with a reason, so the state reflects the
    current regime rather than a fossil.
  - `.claude/sdk/state.py`: `write_state` gained `touch_last_run` (default
    `True`). The skip path passes `False` — stamping `last_run` when the role
    did not run would redefine the field as "last time we considered it" and
    destroy the staleness signal.
  - `.claude/hooks/post-commit`: prints a loud STALE banner naming `/tidy`,
    but only when files are actually queued AND the last real run is failed or
    >= 3 days old. A banner that fires every commit is one nobody reads, which
    is the failure mode being fixed.
  - `.claude/sdk/launch_build.sh`: the completion banner now prints the whole
    post-build DRIVER sequence — tally, commit, sweeps, `/tidy` — and states
    separately that Librarian and Dreamer run IN the DAG and must not be
    re-run after a clean build (with the one exception: a build that stranded
    without committing never landed theirs, which is why they were run by
    hand on 2026-07-28). Previously the banner named only the sweeps.
  - `.claude/commands/tidy.md`: step 5 now says what skipping it costs, and
    that it must be run even when the Tidier could not commit (a collision
    with a live build is a normal outcome; the run still happened).

  **Residual, deliberately not done.** 230 over-length lines remain in
  `cogwheel/tests/**`. Production `cogwheel/lensing/**` is at zero. The test
  files were never in the reflow's scope and hand-rewrapping them is
  high-churn, low-value work that belongs to a Tidier run, not a driver.

  **Patterns worth keeping.** (1) An agent whose state is written only on the
  paths that RUN will fossilize the moment it is disabled, and a disabled role
  is then indistinguishable from a broken one. Record the skip too. (2) The
  Tidier is the ONLY crew role with no automated home — Librarian and Dreamer
  are in the DAG. A single unmechanised step inside an otherwise automated
  pipeline is precisely the one that goes unnoticed, because the pipeline's
  general reliability is what stops anyone from checking.
