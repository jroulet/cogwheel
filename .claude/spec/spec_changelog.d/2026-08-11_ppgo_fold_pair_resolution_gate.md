---
date: 2026-08-11
bump: patch
---
### Pearcey arm: ppGO fast rung fold-pair-existence-or-resolution gate

Updated the `_pearcey_cusp.py` description in the Microlensing engine row
(Key abstractions): the high-w ppGO fast rung's firing conditions now
include a fold-pair-existence-or-resolution gate — a merging min/saddle
fold pair exists (`_merging_fold_pair(...) is not None`) or the node is
geometrically resolved (`w * delta_min >= _PPGO_RESOLUTION_GATE = 4.0`,
mirroring `RHO_END`) — so the geometric-limit serve is never used on an
unresolved node (restores the `select_branch` one-home routing pin).
