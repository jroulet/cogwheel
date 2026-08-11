# Architect Short-Term Observations

### Build: saddle_exterior_full_treatment (2026-08-10)
- Brief: apply full astroid-parity exterior treatment to macro-saddle (gamma>1) exterior
- Professor: cusp-adapted u=d^(2/3) transfers (same d^(-1/3) divergence at deltoid cusps); ghost excision not needed (fold-carrier rescues); parity-gate cusp-window shrink (saddle coverage~0); 1e-3 heldout bar + angular-uniformity test
- Simplifier: ghost excision is test-only today (no production callers); _exclude_near_cusp already parity-agnostic; cusp-adapted u needs interior-anchor generalization (not edge-anchored _wedge_cusp_axis_map); _CUSP_ARM_COVERAGE should be parity-gated with 0.0 placeholder; measurement script post-build; delta ~2 WPs
- Scope: cusp-adapted u for parity==-1 + parity-gated tube shrink constant; stall ghost excision (separate concern)

### Build: ppgo_rung_gate_calibration (2026-08-10)
- Brief: lower _W_PPGO_FLOOR and _R_PPGO_ERROR_CONST so ppGO serves upper portion of excised cusp-window w-band [0.88, 19.3]
- Professor: expected outcome _W_PPGO_FLOOR~7-10, _R_PPGO_ERROR_CONST~2-4; lower edge w~0.88 stays on certified path; calibrate per parity
- Simplifier: 2 WPs (measurement+constants fused); trim script to thin caller; commit script to scripts/; keep _PPGO_BAR_DIVISOR=10
- Scope: upper portion only; acceptance = ppGO serves what certifies, not full [0.88, 19.3] band
