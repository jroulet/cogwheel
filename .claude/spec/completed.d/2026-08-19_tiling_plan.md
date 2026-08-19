---
date: 2026-08-19
section: Lensing training
---

### Engine-free demand-sized tiling plan (order-7a step 2)

`cogwheel/lensing/tiling_plan.py` (`run(...) -> dict`, schema `tiling_plan_v1`, CLI `scripts/tiling_plan.py`) predicts the training campaign's per-`region x parity x gamma_band` tile plan and total engine-call cost. It refreshes the serve-route demand census in demand-only mode (never attaches a surrogate artifact), delegates tile enumeration to the production tilers via the same helpers `tiling_census` uses, and gates each chart tile on positive `engine_residual` demand for its census cell. Every axis is sized to `n = ceil(span / resolution)`: gamma resolution from the caustic-reach derivative, theta from the F083 tube density constant, w from the measured per-cell demand-band edges (clipped at the DD-band ceiling, see below), and the far-field annulus in a declared gauge.

Also fixes Inspector finding INS-1-001 in-build: the w axis was unclipped against the DD-band ceiling in both the measured and prior-box-fallback branches of `_measured_w_range`. Fixed via `_resolve_dd_ceiling`, threaded as `w_ceiling_dd` through `build_plan`/`_plan_region`/`_plan_band` from the census header's `w_band_edges.w_ceiling_dd` (falling back to `chang_refsdal._schwinger.W_CEILING_SCHWINGER` for direct callers), with two new source tags (`measured_clipped_dd`, `prior_box_fallback_clipped_dd`) recording when a clip fired.

The plan is reconciled by three reported (never asserted-fatal) cross-checks — plan nodes vs `surrogate_training._self_estimate`'s blanket-count upper bound, plan nodes vs the `tiling_census` aggregate, and the measured `engine_residual` share vs the `campaign_tiling_design` Fact-1 honest-ledger constant (`_CENSUS_ENGINE_RESIDUAL_LEDGER = 0.4119`) — plus a non-raising escalation verdict that records reasons for the owner to act on.

Engine-free by construction: only the two engine-free predictor siblings (`serve_route_census`, `tiling_census`) are imported; `mpmath` never enters `sys.modules` during a run, asserted by `cogwheel/tests/test_lensing_tiling_plan.py` via `mock.patch` booby-traps on the four wave-amplitude entry points. Inspector verdict: PASS, no new findings.

Order-7a step 2 (demand-sized tiling) is now PLANNED, not yet EXECUTED — this module estimates the campaign, it does not train any chart. `todo.d/lensing_training_campaign` is NOT closed by this entry: steps 3-4 (pre-train checklist, train+attach+7b acceptance) remain open, as does the full acceptance census `todo.d/lensing_no_engine_census` (order-7b).
