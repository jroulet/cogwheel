---
section: Backlog
---
- **The Tidy role has been silently failing since 2026-07-19** `[housekeeping]`
  — `.claude/agent_state/tidy.json` records `last_run: 2026-07-19T03:10:29`
  with `status: "failed"`, and `.serena/memories/tidy_short_term.md` has been
  87 bytes since the same date. Meanwhile the post-commit hook regenerates
  `.claude/tidy_advisory.json` on every commit (last written 2026-07-27
  19:42), so the trigger fires continuously and nothing consumes it.

  Eight days of no-op with no signal. Same failure shape as the rotted test
  suites: a step that reports success by silence, so its absence looks
  identical to its working.

  Worth: (a) finding why the 2026-07-19 run failed (its own logs, or re-run
  it and read the error); (b) deciding whether the advisory trigger should
  fail loudly when nothing consumes it for N commits, rather than
  accumulating; (c) either restoring the role to the post-build sequence or
  retiring it deliberately.

  NOT urgent for correctness — Tidy is repo hygiene, and no defect has been
  traced to its absence. But an agent whose failure is indistinguishable
  from its success is worth either fixing or deleting, not leaving armed.
