---
date: 2026-07-28
---
### Macro-saddle per-lobe interior surrogate charts (lobe-frame serve wired end to end)

The lens-amplification surrogate now serves the macro-saddle's per-lobe
interiors instead of discarding admitted lobe tiles. Each saddle caustic
lobe (of the two disjoint 3-cusp deltoid lobes sitting off the origin on
the shear axis) gets its own frame — source-plane deltoid centroid, the
other lobe's centroid, the inter-lobe corridor half-width, and the
directional boundary `(boundary_theta, boundary_r)` — and is charted in
lobe-local `(rho_lobe, theta_local)` coordinates via the new
`LobeInteriorChart` and `LensAmplificationSurrogate.from_lobe_engine`
(`cogwheel/lensing/surrogate.py`). The directional lobe-boundary radius has
one authoritative definition, `surrogate._lobe_boundary_radius`, shared by
the coordinate maps and the training-side admission test.

Serve-side `_lobe_serves` gates gamma box, log-w band, then the inter-lobe
corridor (`|y - centroid| + corridor_half <= |y - other_centroid|`), then
lobe-local box containment, exclusion balls, image count, and the eta
floor; a source inside the corridor fails the test for both lobes and
falls through to the exact-engine ladder as a named refusal, so no
admitted tile straddles the lobe-equidistance line and no source is ever
served from the wrong lobe. Lobe artifacts carry their own axis-schema tag
(`_LOBE_AXIS_SCHEMA`) and hard-refuse at load under any other tag.
`DATA_CONTRACTS.yaml`'s `lens_amplification_surrogate` entry already
describes per-chart coefficient/knot arrays generically, so no schema bump
is required.

Certified by `cogwheel/tests/test_lensing_surrogate_lobe.py`.
