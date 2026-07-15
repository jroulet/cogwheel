---
bump: minor
---

### Microlensing engine foundation (partial — Build 1 salvage)

New layer `cogwheel/lensing/chang_refsdal/` (IN PROGRESS): double-double
arithmetic substrate (`_dd.py`, 37 tests), exact gauge/cluster-split channel
algebra (`_gauge.py`, 34 tests), and image geometry (`geometry.py` — quartic
solver, delays, magnifications, stationary-phase kernels; tests pending).
Salvaged from the first Build-1 pipeline run after the Professor's inference
review flagged the remaining physics modules (fast complex-1F1 kernel,
contour-free operator, topology-stable channels) as not yet delivered — those
land in Build 1b. Positive-parity macro images only; macro saddles are a
documented limitation.
