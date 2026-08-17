---
date: 2026-08-17
section: Lensing training
---

**Engine-free serve-route demand census (order-7a step 1 of 4)**
`[→ spec]` — build `serve_route_census`. `cogwheel/lensing/
serve_route_census.py` (`run(config) -> dict`, CLI
`scripts/serve_route_census.py`) classifies full-reach lens-prior draws
into seven MECE serve routes via a first-admitting decision waterfall,
tracks per-node route kinds as a D2-reflection-invariant object, and
splits the `engine_residual` population three ways strictly on
`caustic_rho` (never `rho_lobe` — the F073 gauge lesson, pinned by
dedicated regression tests). Engine-free by construction: no
engine-bearing symbol at module top level, engine-adjacent imports
deferred into `_load_production_modules`, called only from `run`.
Demand mode (no `--with-artifact`) asserts zero surrogate-route draws.
HEAD demand report: 83.37% engine_residual, 15.40% born_analytic;
residual split 76.98% interior / 23.02% near-caustic tube, 0%
Born-chart demand. Inspector PASS, no new findings (a prior INS-1-001
dead-code flag on this module confirmed already resolved). SPEC.md
Microlensing-engine row and `spec_changelog.d/2026-08-17_serve_route_
census.md` (minor bump) updated to document the module.

Order-7a steps 2-4 (demand-sized tiling, pre-train checklist,
train+attach+7b acceptance) remain open in
`todo.d/lensing_training_campaign`; the full acceptance census with an
attached surrogate artifact is `todo.d/lensing_no_engine_census`
(order-7b), still pending. Neither todo fragment is closed by this
entry — only the demand-mode census module itself is done.
