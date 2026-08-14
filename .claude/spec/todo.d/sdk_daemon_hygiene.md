---
section: Backlog
---

- **NEXT-SESSION ORDER 1/7 — FIRST, BEFORE ANY BUILD LAUNCH: WIRE DAEMON
  HYGIENE INTO THE BUILD CHOKEPOINTS** `[housekeeping]` — head of the
  queue by driver directive (cheap, no physics, and every later item
  spawns build crews that leak until the watchdog kill-shape is fixed;
  the training campaign is the largest crew of the program). Also apply
  the max_turns lesson recorded in [[lensing_saddle_admission_c3]]. —
  `.claude/sdk/reap_stale_serena.py` shipped 2026-08-14 (project-scoped,
  parent-liveness + age discrimination, dry-run default; born from the
  16-serena/16-pyright accumulation that pinned swap and read as "serena
  hit a complexity threshold"). Remaining wiring, deferred because both
  target scripts were LIVE under a running build:

  1. `launch_build.sh`: pre-launch sweep — run the reap script (`--apply`,
     this project only) and print a loud warning when live same-project
     servers exceed ~3 (the daemon-count analogue of the chart cell-count
     guard: count cheaply at the entry point, never discover as mystery
     latency).
  2. `watchdog.sh` + launcher: the ACTUAL leak — a killed build's crew
     agents orphan their `uv -> serena -> pyright` chains because the
     subtree kill misses grandchildren and SIGKILL reparents them to init
     (measured: yesterday's killed build left a 21-hour 5-server quintet).
     Launch the orchestrator in its own process group (`setsid`) and have
     the watchdog kill the GROUP. MANDATORY after touching either script:
     run `.claude/sdk/verify_watchdog.sh` (~12 s probe), per CLAUDE.md.
  3. CLAUDE.md ops line: run the reap script (dry-run first) before heavy
     work / at session start.

  Multi-project fence is load-bearing: the box also runs
  gw_detection_ias_claude builds with their own serena — the script's
  project discrimination must never widen.
