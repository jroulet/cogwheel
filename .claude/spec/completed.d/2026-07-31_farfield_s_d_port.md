---
date: 2026-07-31
section: Backlog
---

# Far-field `(s, d)` compatibility port (1e-farfield)

Positive-parity `FarFieldChart` construction, selection, persistence, and the
remaining test call sites now use the gamma-resolved fold-adapted `(s, d)`
axes: caustic arc length and signed nearest-fold distance. The required arc
map and axis-schema validation prevent stale caustic-fixed artifacts from
serving.

This completes only 1e-farfield. The 1e-lobe spatial work and the 1e-eta,
1e-w, and 1e-gamma node-measure work remain active. Macro-saddle far-field
charts remain exact-engine fall-through pending their separate per-deltoid-edge
coordinate design.
