---
date: 2026-07-28
bump: minor
---

### Macro-saddle per-lobe interior charts are now servable

The multi-chart narrative previously named only TUBE and FAR-FIELD charts and
described the surrogate as living over origin-centred coordinates. It now also
documents `LobeInteriorChart`: the `gamma > 1` caustic is two disjoint deltoid
lobes off the origin, neither enclosing it, so each lobe carries its own frame
(centroid, other centroid, corridor half-width, directional boundary) and is
charted in lobe-local `(rho_lobe, theta_local)`.

Recorded because this is an ADDITIVE capability, not a correction: the
admission and tiling geometry has existed since S2-2, but every admitted tile
was discarded with `interior_report['served'] = False` because the serve
mapping was strictly origin-centred. The saddle interior was therefore a
STRUCTURAL coverage gap — tiles built, counted, thrown away — rather than a
numerical one. Wiring the lobe frame through the serve path closes it.

Also documents the single-source rule for `r_deltoid`
(`surrogate._lobe_boundary_radius`, shared with
`_SaddleLobeAdmission._r_deltoid`) and the inter-lobe corridor refusal, which
is what guarantees no source is served from the wrong lobe.

`DATA_CONTRACTS.yaml` needs no schema bump: the `lens_amplification_surrogate`
entry describes per-chart coefficient/knot arrays generically, so the added
lobe-frame arrays do not break a contract lock (verified by the Inspector as
INS-3-001).
