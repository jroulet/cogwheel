# Foreman-Lite Short-Term Observations

- INS-1-001 (this session, recurrence 14x+): finding explicitly tagged
  "→ Librarian:" targeting SPEC.md line 56 (tube-chart 4th axis now arc
  length s via theta_to_s map, not raw theta). Declined per role boundary
  — SPEC.md is Librarian-owned, not Foreman-Lite. No files touched. Same
  finding text as prior sessions (was tracked at 13x in long-term memory,
  now 14x+). Per long-term-knowledge guidance this is not a per-pass fix:
  the orchestrator routing bug (dispatching "-> Librarian"-tagged findings
  to Foreman-Lite) needs to be fixed upstream — recommend a pre-filter
  that strips these before they ever reach the Foreman-Lite queue.
