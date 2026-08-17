---
date: 2026-08-17
section: Backlog
---

- **Quiet-phase keepalives now cover the gate waits AND the planning
  turns — the watchdog only kills genuinely dead builds** `[housekeeping]`
  — two measured kills of healthy builds, one defect class (a legitimately
  quiet phase whose log mtime goes stale reads as a wedge):
  (1) escalation wait 2026-08-15/16: `_gate_wait`'s beat backoff doubled
  to an hourly ceiling, so past the third doubling the beat gap exceeded
  the 1200 s watchdog threshold — killed serve_route_census's first launch
  mid-escalation-wait. Fixed: `max_beat` capped at 900 s (300 s margin);
  ~74 beats over an 18 h wait vs the old flood's 270.
  (2) architect planning turn 2026-08-17: the plan-composition turn is one
  long text-only message — no tool calls, so NORMAL-verbosity logging
  writes nothing for the turn's whole duration; the plain `async for`
  planning loops bypassed the inter-message timeout machinery entirely.
  Killed tube_beat_free_representation's recovery launch 1201 s into a
  healthy quiet turn. Fixed: `_iter_query_with_timeout` waits in 240 s
  keepalive slices (bounded by the per-message ceiling, so a genuine
  transport wedge still raises the same TimeoutError), and the architect
  planning/revision loops plus the in-DAG skill runner now route through
  it with `PLANNING_INTER_MESSAGE_TIMEOUT = 3600`.
  Tests: `test_no_beat_gap_ever_exceeds_the_watchdog_threshold` pins THE
  invariant (max beat gap < 1200 s); the old `<25 beats over 18h` pin —
  which asserted the defective backoff — re-pointed. 119/119 SDK tests,
  verify_watchdog probe 20/20.
