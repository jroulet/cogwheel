---
date: 2026-08-14
section: Lensing training
---

**Engine-free tiling census + node-budget predictor (the pre-campaign
gate)** `[housekeeping]` — build `tiling_census_node_budget` + driver
completion (NEXT-SESSION ORDER 6/7). `cogwheel/lensing/tiling_census.py`
(`run(config) -> dict`, CLI `scripts/tiling_census.py`): per
(region x parity) arc/tile/node counts as a THIN CALLER of the production
tilers (fidelity pinned by patching a tiler and requiring the census
count to move by the delta), two-sided bands as report evidence, campaign
call-count cross-checked against `_self_estimate`, and Q1-Q4 answered.
Engine-free import + mock-to-raise pin (the census can never quietly call
the amplitude engine). Inspector PASS, Professor PASS; tree gate RED only
on the part0 absorber guard flagging `_Q4_ASTROID_DD_MARGIN = 60.0` — a
THIRD mirrored copy of `_DD_PRODUCT_MARGIN`. Driver fix: no allowlist
entry; the census reads the production constant at its use site (the
module defers its `surrogate_training` import, so the read lives in
function scope) and the test oracle reads the same production constant.
Census suite + part0: 44/44 green; tree green by union (delta since the
build's gate is the DRY re-point only).

The guard has now fired on all three of its design classes in one day:
a new measured-margin constant (allowlist with justification, c3 build),
a dead-units constant retirement (F079 build), and a mirror-drift copy
(this build — resolved by DRY, not allowlisting). It is earning its keep.
