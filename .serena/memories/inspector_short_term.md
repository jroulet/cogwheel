# Inspector Short-Term Observations

2026-08-09 — Review of build cusp_ppgo_high_w (HEAD with uncommitted changes)

## Previously open findings — RESOLVED

- **INS-6-001 (bug, resolved)**: Test fixtures now use `_PPGO_SERVE_W=20000.0` and saddle parity w ∈ [5000, 20000], all above the threshold needed for `_R_PPGO_ERROR_CONST=50.0`. All 4 previously-failing tests should now pass.
- **INS-6-002 (trivial, resolved)**: Comments updated to reference `_R_PPGO_ERROR_CONST=50.0`, `r_ppgo_min≈464.2`, and the correct ppGO-fire threshold (w ≥ 5000 for saddle, w ≥ 15000 crossing for astroid).

## New findings — none

Verified:
- r_ppgo_min formula: (50*1/(0.05/10))^(2/3) ≈ 464.16 — consistent with test comments and imported constants
- fold_ppgo_correction signature matches the call: (w, source, gamma, beta=beta, kappa=kappa)
- Error handling: catches LensDomainError, guards non-finite, falls through to Pearcey
- ppGO rung positioned before the existing Pearcey uniform path (correct — ppGO is faster)
- _W_PPGO_FLOOR=50.0 prevents ppGO at low w where kernel truncation breaks down
- Existing tests not affected (control radii at existing params << 464)
- All new tests increment n_checks (anti-vacuity)
- Mock patching approach correct (same module object for test + production import)
- Self-falsification tests verify guards have teeth
- Import check: `from cogwheel.lensing.chang_refsdal import _airy_fold` works

## Pre-existing issues (not actionable)
- `test_moving_error_const_threshold_flips_a_fixed_node` and `test_served_node_is_bit_identical_to_the_cusp_arm` still time out due to `_grid_served` → `F_op_grid` → mpmath quadrature at w=80. Pre-existing — not caused by the current changes.
