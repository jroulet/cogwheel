---
date: 2026-08-17
bump: minor
---

Engine-free serve-route demand census shipped
(`cogwheel/lensing/serve_route_census.py` + CLI
`scripts/serve_route_census.py`, order-7a step 1 of
`todo.d/lensing_training_campaign`): classifies lens-prior draws into
seven MECE serve routes via a first-admitting waterfall, tracks
per-node route kinds as a D2-invariant object, and splits the
engine_residual population on `caustic_rho` (F073 gauge guard).
Demand mode asserts zero surrogate draws; HEAD demand report recorded
in SPEC.md. Documentation-only sync — no behavior change.
