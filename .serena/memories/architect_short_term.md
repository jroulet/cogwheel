# Architect Short-Term Observations

2026-08-08 — Plan for build cusp_ppgo_high_w:
- Gating on control radius R (not w) per Simplifier — R is the correct asymptotic parameter, composable with envelope_bar, source-independent.
- Professor confirms: Pearcey → geometric image sum as (x,y) → ∞; fold_ppgo_correction converges to same limit; both branches (astroid+saddle) valid; bar_ppgo = envelope_bar/10 for calibration target.
- Dual gate: R ≥ r_ppgo_min AND w ≥ w_floor (kernel-truncation guard).
- Provisional constants: _R_PPGO_ERROR_CONST=2.0, _W_PPGO_FLOOR=50.0, _PPGO_BAR_DIVISOR=10.
- DO-NOTHING: fold_ppgo_correction already has internal guards; LensDomainError caught → fall through to Pearcey.
- Post-build: driver measures and tightens _R_PPGO_ERROR_CONST.
