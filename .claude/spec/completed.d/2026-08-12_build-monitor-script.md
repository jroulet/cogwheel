---
date: 2026-08-12
section: Backlog
---

### One build monitor, three blind spots closed

Closes `todo.d/sdk_killing_a_build_leaves_its_monitor_armed.md`, and takes the
two sibling defects with it.

The driver hand-rolled its progress-monitor pipeline inline at every launch.
Three distinct authoring bugs accumulated, each costing real build time:

1. **SELF-MATCH.** The harness echoes the Monitor's own command text into the
   build log, and that text contains every marker the filter greps for. An
   unfiltered watcher matches itself on the first poll.
2. **DEDUPE ON REPEAT** (measured 2026-08-12). Comparing the newest matching
   line against the last emitted one is blind to a byte-identical repeat.
   `[file-based]` lines carry NO timestamp, so the second
   `Plan written to <dir>/plan.json` after a rejected-and-resubmitted plan is
   identical to the first. The monitor stayed silent through it and the build
   waited 8 minutes on driver approval with the watchdog staleness clock
   running. At 2400 s it was never at risk; a tighter threshold would have
   killed a healthy build over the watcher's blind spot.
3. **NO TEARDOWN** (measured 2026-08-06, the fragment this closes). Killing or
   relaunching a build does not stop its monitor, so a watcher sits on a dead
   log reporting stalls that mean nothing — and a watcher on a dead log can
   never reach its terminal condition, making it permanently indistinguishable
   from a hung build. Four accumulated in one session.

The common cause is NOT "the watcher cannot distinguish two identical states"
— that framing is loose enough to describe any bug, and the three have
different mechanisms and different fixes. It is that each watched an
easily-observed PROXY that is many-to-one onto the quantity actually needed:
last-line text for occurrence count (2), log growth for process liveness (3).

**Fix.** `.claude/sdk/build_monitor.sh <log> [poll=120] [build_pid]` encodes
all three once: it strips `Monitor(persistent` lines, emits every NEW
occurrence by tracking the COUNT of matching lines (capped per burst so a
chatty stretch cannot trip the harness rate limit), announces a stall ONCE on
entering it, exits 0 on a terminal marker, and exits 0 when the build is gone.
The optional PID is strongly preferred: without it the liveness check matches
ANY `sdk/build.py`, so a second unrelated build keeps the monitor alive at a
dead log — the exact failure it exists to prevent. The teardown check needs no
cooperation from whoever does the killing, which is where the discipline
actually failed.

`launch_build.sh` now prints the exact invocation (log + PID) under
"ARM THIS MONITOR -- do NOT hand-roll one", replacing the old
`tail -20 <log>` hint that left the driver to reconstruct the pipeline.

**Verified by falsification**, each blind spot on the real line format:
a log seeded with a `Monitor(persistent` echo containing `Build failed` and
`GATE FAILURE` did NOT false-terminate (1); a byte-identical `Plan written`
repeat WAS emitted where last-line comparison went silent (2); a dead PID made
it exit 0 rather than report stalls, while a real build was concurrently
running and would have kept the generic `pgrep` check alive (3).

A first attempt at (2) used synthetic TIMESTAMPED lines, which are not
byte-identical, so it failed to reproduce the bug at all and would have
validated a no-op fix. Reproduce the miss on the real line format before
believing a monitor fix — recorded in AGENTS.md alongside the rule.
