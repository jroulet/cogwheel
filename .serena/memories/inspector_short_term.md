# Inspector Short-Term Observations

## Review: 2026-08-03 (pass 7) — DD w-ceiling + arc-length axis (re-review, unchanged diff) — PASS

### Scope
Re-review of uncommitted changes to `cogwheel/lensing/surrogate.py` (43 lines added/modified) and new untracked test file `cogwheel/tests/test_lensing_wedge_dd_arclength.py` (20 tests). Diff is byte-identical to pass 6.

### What Changed (unchanged from pass 6)
- New module-level constant `_DD_PRODUCT_MARGIN = 58.0` (line 125), matching `surrogate_training._DD_PRODUCT_MARGIN`.
- `from_wedge_engine` restructured: `_log_w_grid()` call moved AFTER the DD cap computation.
- DD ceiling computation: `theta_mask` on wedge_map.theta_nodes within theta_wedge_range, `reach_max = max(r_table[:, theta_mask])`, `dd_w_cap = 58.0 / (r_grid[-1] * reach_max)`, `w_range` capped.
- Arc-length map: `rep_gamma = median(gamma_grid)`, `arc_theta_fine = linspace(theta_wedge_range[0], ..., 2001)`, `caustic_speed(..., branch=1)`, `cumulative_trapezoid(...)`, `theta_to_s = vstack(...)`, `s_grid = np.interp(theta_wedge_grid, ...)`.
- Updated `from_wedge_values(...)` call to pass `theta_to_s=theta_to_s, s_grid=s_grid`.
- New test file with 4 test classes (DDWCeilingTestCase, ArcLengthAxisTestCase, NoDDCapLowWTestCase, SelfFalsificationTestCase) = 20 tests total.

### Correctness Re-Assessment
- **DD formula**: Correct. `w_max * r_max * reach_max <= 58` guaranteed. Conservative (uses worst-case over all gamma/theta in range). Edge case (cap < w_min) raises ValueError cleanly from `_log_w_grid`.
- **Brief vs code**: Brief erroneously suggests `DD_MARGIN / (r_min * reach_max)` but this is the LEAST conservative cap. Code correctly uses `r_grid[-1]` (r_max) for the tightest global bound. Correct deviation from the brief.
- **Arc-length map**: Correct. `arc_theta_fine` and `theta_wedge_grid` share exact endpoints (both from `theta_wedge_range`), so no extrapolation in `np.interp`. `s_grid` nodes are exact images of `theta_wedge_grid` through the same map. `branch=1` correct for positive-parity interior.
- **Serve-time plumbing**: Already existed — `_evaluate_chart`'s InteriorWedgeChart branch correctly checks `theta_to_s is not None` and remaps `theta_wedge -> s` via `np.interp`.
- **NPZ persistence**: Existing code already saves/loads `theta_to_s` for wedge charts.
- **Backward compatibility**: `from_wedge_values` accepts `theta_to_s=None, s_grid=None` defaults — existing tests (40) pass unchanged.
- **All 20 new tests PASS** (76s).
- **All 40 existing wedge tests PASS** (39s).
- **Import check**: `from cogwheel.lensing.surrogate import _DD_PRODUCT_MARGIN` — OK.

### Findings (trivial only — carried forward)

1. **INS-w3-001 (trivial, still open)**: Local variable `_ARC_MAP_NODES = 2001` at line 3786 duplicates the value of module-level `_FARFIELD_ARC_MAP_SIZE = 2001` (line 151). Could reference it directly. Comment notes the match; harmless.

### Open Issues Carried Forward (pre-existing, not from this diff)
- INS-w2-001 (trivial): Stale comments in test_lensing_interior_wedge_chart.py (lines 882, 1039) referencing a "bug" that was already fixed.
- INS-w-004 (design — Librarian scope): DATA_CONTRACTS.yaml does not describe InteriorWedgeChart.
- INS-w-005 (design — Librarian scope): SPEC.md does not mention InteriorWedgeChart.
- INS-1-001 (trivial): Unreachable `C <= 0.0` guard in ppgo_map.py.
- INS-1-002 (trivial): DATA_CONTRACTS empty-range semantics.
- INS-1-003 (trivial): Misleading `_EXTRAP_W_CERT_DEFLATION` name.
