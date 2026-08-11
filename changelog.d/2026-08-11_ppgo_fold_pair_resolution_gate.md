---
date: 2026-08-11
---
### ppGO fast rung gated on fold-pair existence or geometric resolution

The cusp arm's high-w ppGO fast rung (in
`cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`, `cusp_amplification`)
now serves `_airy_fold.fold_ppgo_correction` — the geometric-image-sum
limit of the Pearcey function — only when a merging min/saddle fold pair
exists (`_merging_fold_pair(...) is not None`) or the node is
geometrically resolved (`w * delta_min >= _PPGO_RESOLUTION_GATE = 4.0`,
mirroring `RHO_END`).  Previously an unresolved node with no fold pair
could still be served a value bit-equal to `geometric_amplification`,
silently disagreeing with the `select_branch` wave/geometric predicate and
flipping the one-home routing pin (`test_thresholds_have_one_home`).
The None-fall-through contract is unchanged: a rung refusal falls to the
uniform Pearcey path.  A self-falsification test
(`test_resolution_gate_isolated_admit_and_refuse` in
`test_lensing_airy_fold.py`) isolates the gate on a no-fold-pair saddle
fixture: intact gate admits, inflated gate blocks, disabled gate admits,
resolved `w` admits even under the inflated gate.
