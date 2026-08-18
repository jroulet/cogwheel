---
bump: patch
---

### Born trained-floor band split (direction (a) shipped)

`_born_residual_analytic` gained Route 2: when the residual chart's box
covers the host but the host sub-band's low edge drops below the chart's
trained `log_w` floor, the chart now serves the trained sub-band
`[trained_floor, w_trust]` and the exact engine hosts the untrained
remainder below it, instead of refusing the whole band to the engine.
`serve_route_census.py`'s engine-free census mirrors the new route via
`_born_trained_floor_route`. In-build 10k census (n_freq=8) recovers
3.43% of draws to `born_analytic` (previously `engine_residual`/
`diffractive_analytic`). Direction (b) (corrected-carrier far-field
serve) remains superseded by the separate two-image GO carrier
direction; `lensing_born_farfield_completion` stays open.
