---
date: 2026-08-17
bump: minor
---

Serve-route census corrected to mirror the production band ladder
(audit): arms offered only for 60 < w <= 150, select_branch only above
150, and above-ceiling both-arms-decline nodes are `refused` — new
`wave_refused` route (8-label MECE), excluded from residual_demand.
Corrected 10k demand map: 72.25% residual / 15.40% Born / 12.03%
wave_refused / 0.32% saddle_c3. Driver-side fix per census audit;
suite 34/34.
