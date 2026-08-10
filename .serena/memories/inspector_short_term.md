## 2026-08-10 — Review: Exterior 2D (rho, u) fold-carrier (re-review, final)

Re-reviewed the 2D fold-carrier build. All tests pass.

### Test results (all green)
- test_lensing_exterior_polar_fold.py + test_lensing_surrogate_training.py: 164 passed, 76 skipped, 0 failed

### Verified correct (no new issues found)
- `_probe_ghost_delay`: factored-out single-point ghost probe, returns None on error, correctly handles matrix parameter from caller
- `_compute_rho_u_carrier`: 2D `(n_rho, n_theta_c)` ghost-kernel probing, median over gamma, NaN fill along u (axis=1) then rho (axis=0) with linear interp + zero-order hold at boundaries
- `from_values` demodulation: `exp(-1j * w * rho_u_carrier[:,:])` broadcast shape `(n_w, 1, n_rho, n_theta_c)` × `env_complex` shape `(n_w, n_gamma, n_rho, n_theta_c)` — correct
- `_evaluate_chart` re-modulation: bilinear interpolation in (rho, u) using searchsorted + clamped brackets, re-modulates `exp(+1j * w * carrier_val)`. Uses raw `rho` (not log-transformed v1). Correct.
- `_chart_to_npz` writes V5 schema and `rho_u_carrier` key (conditional)
- `_chart_from_npz`: backward compat via `data.get('rho_u_carrier')` then `data.get('rho_carrier')` → `np.broadcast_to` 1D→2D
- Schema V4+V5 both in known set; `_chart_to_npz` writes V5, `_build_provenance` writes V5
- No stale `_compute_rho_carrier` / `rho_carrier` references in production code (only in backward-compat load key at surrogate.py:4428)
- No stale `_compute_rho_carrier` references in tests — all mocks use `_compute_rho_u_carrier`
- Mock `zero_carrier` shape is `(n_rho, 4)` = `(n_rho, n_theta_c)` — correct 2D
- Import probes pass; all changed suites green
- `from_engine`: continuity gate + k_chart both use the 2D-demodulated envelope when `fold_carrier=True` and `rho_u_carrier is not None`

### STILL OPEN (Librarian scope — doc staleness, NOT code defects)
- **INS-1-002**: SPEC.md ~line 63 says `exterior_polar_rho_log_carrier_v1` is "the ONLY known tag" and describes `rho_carrier` as 1D `(n_rho,)`. Still stale — V5 is now also known and carrier is 2D.
- **INS-1-003**: DATA_CONTRACTS.yaml line 199 says `axis_schema='exterior_polar_rho_log_carrier_v1'` is "the ONLY known tag" and describes `rho_carrier` as 1D `(n_rho,)`. Still stale — not updated in this build.

### New findings: NONE
No new code bugs, design issues, or test gaps found.
