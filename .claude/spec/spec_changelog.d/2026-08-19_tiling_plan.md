---
date: 2026-08-19
bump: minor
---

Engine-free demand-sized tiling plan shipped (`cogwheel/lensing/tiling_plan.py` + CLI `scripts/tiling_plan.py`, order-7a step 2 of `todo.d/lensing_training_campaign`): `run(...)`/`build_plan(...)` predict the training campaign's per-`region x parity x gamma_band` tile plan and total engine-call cost by refreshing the serve-route demand census and delegating tile enumeration to the production tilers exactly as `tiling_census` does, gating each chart tile on positive `engine_residual` demand and sizing every axis to `n = ceil(span / resolution)`. Also fixes INS-1-001 (DD-band w-axis ceiling clip) via `_resolve_dd_ceiling`, threaded through `build_plan`/`_plan_region`/`_plan_band` with two new source tags recording when a clip fired. Zero wave-optics evaluations; three cross-checks and an escalation verdict are reported, never asserted-fatal.
