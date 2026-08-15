---
date: 2026-08-14
bump: minor
---

Engine-free tiling census + node-budget predictor shipped
(`cogwheel/lensing/tiling_census.py` + CLI): thin caller of the
production tilers, two-sided bands, campaign call-count estimate
cross-checked against `_self_estimate`, pre-campaign questions Q1-Q4.
Build `tiling_census_node_budget` + driver DRY fix (the Q4 DD-margin
mirror replaced by reading the production `_DD_PRODUCT_MARGIN` at the
use site — the part0 absorber guard caught the third mirrored copy).
