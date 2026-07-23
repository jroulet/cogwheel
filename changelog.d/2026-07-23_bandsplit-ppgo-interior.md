---
date: 2026-07-23
---
### Band-split serving, certified-ppGO map, interior tiles, subdivision (Build 8h-a)

The lensed likelihood now serves each draw's frequency band per-node:
chart-served below the certified-ppGO trust floor, bare ppGO (the image
kernel sum) above it, with the floor read from a new hash-pinned
``certified_ppgo_map`` data product measured against exact references
(sup-over-w certification, worst-of-five-angles per cell, F-normalized
1e-4 bar, margin rule ``max(1.5 w_cert, w_cert + 2)``). Draws whose band
crosses the parity's Schwinger wall are never band-split (beyond-wall
certification does not exist and never extrapolates). The trainer gains
an interior (4-image) far-field tile family (admission keyed off caustic
topology by winding number — the saddle's two off-origin deltoid lobes
admit no origin-centred interior and record a loud skip), ppGO-aware
strata trimming, and single-level subdivision of eps-gated edge tiles
with per-child admission and disposition records.
