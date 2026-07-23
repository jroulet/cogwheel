---
bump: patch
---

### ppGO map truncation-on-refusal: per-cell w_ceiling and rho_measured_max

`certified_ppgo_map` schema bumps to `0.2.0`. Each cell now carries a
measured `w_ceiling` (min over angles of that angle's bisected max
accepted `w`: a monotone saddle-image-branch refusal part-way up a
cell's `w` sweep truncates that angle at its accepted w-prefix instead
of invalidating the whole cell — the cell certifies on its measured
range, trusted only on `[w_cert, w_ceiling]`) and `rho_measured_max`
(the open outer annulus `[4.0, inf)` is sampled at one finite radius,
so every cell records how far out that sample reached; accessors
return UNKNOWN beyond it). The provenance scalar gains `w_ceiling_rule`
and `rho_measured_max_rule`, and the SHA1 content hash now covers both
new grids. `CertifiedPpgoMap.load` hard-refuses (`KeyError` ->
`use_certified_ppgo_map` returns `False`) any map missing `w_ceiling`
or `rho_measured_max`.

Consumers updated to match: the band-split dispatch in
`cogwheel/lensing/likelihood.py` now places the ppGO-vs-chart split at
`min(parity_wall, w_ceiling)` instead of the wall alone, and
`cogwheel/lensing/surrogate_training.py` strata trim only trims a
stratum when the cell's ceiling covers that stratum's top edge.
