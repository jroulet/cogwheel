---
section: Backlog
---

- **THE PLAN GATE HAS NO WIDTH CHECK** `[housekeeping]` — 2026-07-30. AGENTS.md
  says to reject over-wide plans at the plan gate, but nothing computes width,
  so the check happens only if the reviewer remembers to do it by eye while
  absorbed in correctness. On 1e-farfield the driver reviewed the plan hard for
  correctness (catching four real defects) and never checked width at all.

  A working checker exists and is calibrated; it needs to move into
  `.claude/sdk/plan_width_check.py` and be named in the launch banner beside
  the approval instructions.

  It reports: WP count vs the ~3 ceiling, per-WP `how` size and file count,
  how many domain-test specs name a suite file vs are substantive-but-unscoped
  (the F057 shape), the PREDICTED shard count and per-shard turn budget, and
  whether the `test_dev` node will even run.

  ## Two traps it must keep documenting

  1. **`plan.json` is NOT JSON at the gate.** `gates.py:372` writes the
     human-readable MARKDOWN summary there; `orchestrator.py:929` overwrites it
     with real JSON only AFTER approval, for crash recovery. A checker written
     against a post-approval file parses cleanly in testing and fails at the
     one moment it is useful — the F058 shape again.
  2. **Do NOT reject on `how` length.** Measured: in the plan that died, WP1
     was 3121 chars and SUCCEEDED ($14.34, 20 min) while WP2 was 3178 chars and
     exhausted. Length would have passed the WP that died and rejected the one
     that lived. It stays advisory; a check that fires on healthy plans trains
     the reader to ignore the channel.

  ACCEPTANCE: run against the two archived plans and show it flags the
  substantive-but-unscoped count on the one that died while passing the one
  that ran; wire it into the launch banner.
