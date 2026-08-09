---
date: 2026-08-09
bump: patch
---

### High-w ppGO fast rung in the cusp (Pearcey) arm

The `_pearcey_cusp.py` description gains the high-w ppGO fast rung in
`cusp_amplification`: when the control radius clears the ppGO gate (the
uniform-form leading-error bar, tightened by the `_PPGO_BAR_DIVISOR` divisor
against the `_R_PPGO_ERROR_CONST` coefficient, with `w >= _W_PPGO_FLOOR`) it
serves `_airy_fold.fold_ppgo_correction` directly — the geometric-image-sum
limit of the Pearcey function — a ~10^3x faster path that returns before any
table or quadrature lookup; the None-fall-through contract is unchanged
(rung refusal falls to the uniform Pearcey path). The rung serves both the
astroid (positive-parity) and saddle cusp branches.
