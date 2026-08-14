---
date: 2026-08-14
section: SDK pipeline
---

**Daemon hygiene wired into the build chokepoints** `[housekeeping]` —
commit `156dbab` (NEXT-SESSION ORDER 1/7). `launch_build.sh` now runs
`reap_stale_serena.py --apply` (this project only) pre-launch and warns
when live same-project serena instances exceed 3 (`--count-live`, counted
by chain root — no duplicated discrimination in bash). The orchestrator
launches under `setsid`; `watchdog.sh` kills the process GROUP when the
orchestrator leads its own group (fallback: old parent-walk for hand
launches), closing the 2026-08-13 orphaned `uv -> serena -> pyright`
leak. `verify_watchdog.sh` gained test 8 reproducing the leak shape
(setsid orchestrator + grandchild; grandchild must die): 20/20 pass.

Semantics fix found while wiring: `reap_stale_serena.py` verdicts are now
per-CHAIN at the root — judging members separately made an orphaned
wrapper/server pair immortal (wrapper kept for its live child, child kept
for its live parent). The c33e15f live-child proxy is replaced by the
signal it stood in for: an ESTABLISHED TCP peer ("actively serving").
Measured on the live box: the 27 h port-8323 pair is genuinely serving a
live claude client and stays kept; nothing is falsely reapable even at
`--min-age-hours 0`. AGENTS.md carries the ops line (dry-run at session
start / before heavy work; never widen project discrimination).
