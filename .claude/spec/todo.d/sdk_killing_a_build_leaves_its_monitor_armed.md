---
section: Backlog
---

- **KILLING A BUILD DOES NOT STOP ITS MONITOR, SO DEAD LOGS KEEP FIRING FALSE
  STALLS** `[housekeeping]` — the launch protocol pairs launch with arming a
  progress monitor (MECHANICAL PAIRING, in AGENTS.md). There is no matching
  pairing on the way down. A killed or relaunched build leaves its monitor
  watching a log that will never advance again, which then reports a stall
  that means nothing.
  MEASURED (2026-08-06/07): four stale monitors accumulated in one session,
  each firing against a dead log, at one driver invocation per emission. One
  of them was still reporting on the FIRST launch's log after the build had
  been killed and relaunched — so the driver was reading progress for a run
  that no longer existed while the live run went unwatched.
  This compounds a hazard already recorded in memory: a stalled monitor and a
  finished one look identical, so the guidance is to prefer a watcher that
  EXITS on its terminal condition. A watcher on a dead log can never reach
  its terminal condition, so it is the worst case of that shape — permanently
  indistinguishable from a hung build.
  FIX: make teardown symmetric with launch. Either (a) the kill path stops the
  monitor for that log, same action, never two; or (b) monitors self-terminate
  when the build process for their log is gone — the terminal check already
  greps `pgrep -f "sdk/[b]uild.py"`, so a monitor can exit on "log frozen AND
  no build process" instead of reporting a stall forever.
  (b) is strictly better: it needs no cooperation from whoever does the
  killing, which is where the discipline actually failed.
