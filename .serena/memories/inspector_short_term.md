## 2026-08-10 — Re-review: exterior fold-carrier phase demodulation (INS-1 fixes)

Re-reviewed the latest revision of the fold-carrier build. Coder addressed all three INS-1 findings from the previous review pass.

### RESOLVED
- **INS-1-001**: `_evaluate_chart` fold-carrier re-modulation at line ~2893 now correctly uses `np.exp(log_w_clamped)`. Verified by re-reading the source and grepping for `np.exp(log_w_query)` — zero occurrences.
- **INS-1-002**: `GhostExcludedTilesInRegionReportTestCase.test_ghost_excluded_tiles_is_zero` now asserts `ghost_ct == 0`. The mock on `_exclude_ghost_dominated` is dead but harmless (the code path was removed, the mock is never invoked).
- **INS-1-003**: All 6 test methods in `test_lensing_surrogate.py` updated from `exterior_polar_rho_log_v3` to `exterior_polar_rho_log_carrier_v1`. New `test_rho_log_v3_schema_raises_valueerror` test added to `ExteriorPolarStaleSchemaHardRefusalTestCase`.

### STILL OPEN (Librarian scope — not induced by this revision)
- **INS-1-004**: SPEC.md line 62 still says `'exterior_polar_rho_log_v3'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V3`); code uses `'exterior_polar_rho_log_carrier_v1'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V4`).
- **INS-1-005**: DATA_CONTRACTS.yaml line 199 still says `axis_schema='exterior_polar_rho_log_v3'`.

### Test results (all green)
- test_lensing_exterior_polar_fold.py: 41 passed
- test_lensing_surrogate.py (schema tests): 27 passed
- test_lensing_surrogate_training.py (fast tier): 106 passed, 67 skipped (training tier), 9 deselected

### Verified correct (re-confirmed)
- INS-1-001 fix: `_evaluate_chart` uses `log_w_clamped` for fold-carrier re-modulation
- INS-1-002 fix: test asserts `ghost_ct == 0`
- INS-1-003 fix: all schema tags updated, V3 refusal test added
- Composition order: fold-carrier demod BEFORE carrier_rate in `from_values`; carrier_rate remod BEFORE fold-carrier remod at serve
- `_chart_to_npz` / `_chart_from_npz` rho_carrier round-trip correct (optional field, `.get()` fallback)
- `_build_farfield_chart` / `_train_band_charts` fold_carrier threading correct
- All consumer callers of `_farfield_exterior_tiles` backward-compatible (gamma_band + ghost_drop_count retained in signature)
- Import chain clean
- No new blocking findings
