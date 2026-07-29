---
date: 2026-07-29
---

### Retired the legacy microlensing operator series; added a distance-to-caustic admission leg to the geometric branch

The shear-free (`gamma' == 0`) point-lens exit in the Chang-Refsdal engine no
longer runs the legacy dd/1F1 operator-series contraction. It now serves a
closed form instead, byte-identical to the retired series: at zero shear the
operator that the series expanded is the identity, so the series collapsed to
its zeroth term, which is exactly the point-mass kernel already computed
elsewhere in the engine. The legacy contraction remains available only for
test-only oracle duty. `CancellationError`, which was raised only inside that
contraction, is removed.

Separately, the geometric branch's admission gate (`select_branch`) gains a
third condition: sources must additionally be at least `ETA_MIN_GEOMETRIC =
0.3` from the lensing caustic to be served by the geometric-optics asymptote.
Measurement (FINDINGS F031) showed the previous two-condition gate admitting
nodes near the caustic with up to 117% relative error; the new leg cuts
worst-case error to 7.65e-5 at the cost of routing more near-caustic nodes to
the uniform-asymptotic arms or a named refusal instead.
