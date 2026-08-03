---
bump: patch
---

Sync Born rung documentation with C8/C11 code changes: update region description
from prior-box annulus to caustic-relative exterior (`rho > 1`), correct
`born_gate` guard count from three to two (parity-split exterior fences
retired in C8, F036), update census description, fix serve-slot status
(wired since C11). Add `ppgo_map.py` and `born_residual_chart.py` to
module table row 55.
