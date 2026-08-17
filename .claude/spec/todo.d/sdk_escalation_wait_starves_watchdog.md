---
section: Backlog
---

- **THE ESCALATION WAIT'S KEEPALIVE BACKS OFF PAST THE WATCHDOG
  THRESHOLD — a healthy waiting build gets killed** `[housekeeping]` —
  measured 2026-08-15 (serve_route_census, first launch): escalation
  written 08:46; the wait loop's "still waiting" log lines appeared at
  4 m, 12 m, 28 m (exponential-ish backoff), then nothing; the watchdog
  killed the build at 09:34 — log stale 1201 s — while it was alive and
  legitimately waiting for a driver decision that could not arrive (the
  driver was down on a spend limit). Same defect class as the silent
  tree gate (fixed 2026-08-14 with the Popen heartbeat): a healthy
  long-running phase whose log goes quiet reads as a stall. FIX: cap the
  escalation-wait keepalive interval well under the staleness threshold
  (e.g. print every 300 s, or touch the log mtime), in gates.py's
  file-based wait loop. Do NOT lengthen the watchdog threshold instead —
  real stalls must still die. Apply the verify-watchdog discipline after
  touching the wait loop (extend the probe if it can cheaply simulate an
  escalation wait).
