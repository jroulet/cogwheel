# Architect Short-Term Observations

### Build: ppgo_rung_gate_calibration (2026-08-10)
- Brief: lower _W_PPGO_FLOOR and _R_PPGO_ERROR_CONST so ppGO serves upper portion of excised cusp-window w-band [0.88, 19.3]
- Professor: expected outcome _W_PPGO_FLOOR≈7-10, _R_PPGO_ERROR_CONST≈2-4; lower edge w≈0.88 stays on certified path; calibrate per parity
- Simplifier: 2 WPs (measurement+constants fused); trim script to thin caller; commit script to scripts/; keep _PPGO_BAR_DIVISOR=10
- Scope: upper portion only; acceptance = ppGO serves what certifies, not full [0.88, 19.3] band