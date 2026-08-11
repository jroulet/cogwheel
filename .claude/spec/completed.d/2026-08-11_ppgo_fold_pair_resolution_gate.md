---
date: 2026-08-11
section: Backlog
---
## ppGO fast rung gated on fold-pair existence or resolution — one-home pin GREEN

Resolves `lensing_one_home_routing_disagreement` (option (a), the
resolution guard). Code in working tree,
`cogwheel/lensing/chang_refsdal/_pearcey_cusp.py` (`cusp_amplification`).

- **Dual gate on the ppGO fast rung**: the rung now serves
  `_airy_fold.fold_ppgo_correction` (the geometric-image-sum limit of the
  Pearcey function) only when a merging min/saddle fold pair exists
  (`_airy_fold._merging_fold_pair(images, source, matrix) is not None`) OR
  the node is geometrically resolved (`w * delta_min >=
  _PPGO_RESOLUTION_GATE = 4.0`, mirroring `operator.RHO_END` — mirrored
  because `operator.py` imports `_pearcey_cusp` at module level, creating a
  circular import).  Otherwise the rung refuses (`result = None`) and falls
  through to the uniform Pearcey path, unchanged in contract.  Rationale:
  `fold_ppgo_correction` is the geometric limit, so serving it on an
  unresolved node (`w*delta_min < 4.0`, no fold pair) returned a value
  bit-equal to `geometric_amplification` and silently flipped the one-home
  routing pin (`select_branch` said 'wave' but the grid served 'geometric').
- **Test**: `test_resolution_gate_isolated_admit_and_refuse` in
  `PpgoRungSelfFalsificationTestCase` (`test_lensing_airy_fold.py`) isolates
  the gate on a saddle fixture with two saddle-type images (no fold pair, so
  the resolution leg decides alone): intact gate admits at `w = 500`
  (`w * delta_min` ≫ 4.0), inflated gate (1000) blocks, disabled gate (0)
  always admits, and a resolved `w = 20000` admits even under the inflated
  gate.

ACCEPTANCE (all met): `test_thresholds_have_one_home` passes with a
non-zero comparison count (verified green); the chosen option is the
physics-justified resolution guard (option (a)); `test_lensing_operator.py`,
`test_lensing_fast_path.py`, `test_lensing_airy_fold.py` green (Inspector
verdict PASS); no regression in the eta-leg-live assertion.  The last
remaining red in `lensing_serving_ladder_guards_are_red` is now resolved —
that fragment is fully retired.
