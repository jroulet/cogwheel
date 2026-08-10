## 2026-08-10 — Review: carrier_demod follow-up (re-review of uncommitted working tree)

Reviewed the full uncommitted diff on branch `claude-dev`. The working tree implements carrier demodulation for ExteriorPolarChart per the approved plan.

### Verified correct
- Schema bump: `_EXTERIOR_POLAR_AXIS_SCHEMA` → `_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER` (`'exterior_polar_rho_u_v1'` → `'exterior_polar_carrier_demod_v2'`)
- Old schema `exterior_polar_rho_u_v1` dropped from known set → hard-refuses at load (tested)
- Old schema `exterior_polar_rho_theta_c` also dropped (pre-existing, tested)
- `carrier_rate: float = 0.0` on ExteriorPolarChart (default backward-compatible)
- Single canonical demodulation site: `from_values` (when carrier_rate != 0, applies `exp(-1j * k * w)` before fitting)
- Remodulation in `_evaluate_chart`: only ExteriorPolarChart, only when carrier_rate != 0, `exp(+1j * k * w)` after spline contraction
- from_engine estimates k_chart from raw envelope: per-node unwrap along w, median of valid nodes
- _assemble validates `np.isfinite(carrier_rate)` → raises ValueError
- NPZ write: _chart_to_npz writes carrier_rate
- NPZ load: _chart_from_npz reads via `meta.get('carrier_rate', 0.0)` for backward compat
- No stale `_EXTERIOR_POLAR_AXIS_SCHEMA` (without _CARRIER) references in source code
- From_values signature backward-compatible: carrier_rate defaults to 0.0
- All 92 surrogate tests pass, 99 farfield+lobe tests pass (27 skipped)
- Import chain clean

### NEW finding
- **INS-17-002**: `n_w = log_w_grid.size` at surrogate.py line ~2999 (in LensAmplificationSurrogate.from_engine's k-chart estimation loop) is defined but never used — dead code. [trivial]

### STILL OPEN
- **INS-16-002**: SPEC.md line 63 still says `'exterior_polar_rho_u_v1'` — should be `'exterior_polar_carrier_demod_v2'`. Librarian scope.
- **INS-16-003**: DATA_CONTRACTS.yaml line 199 still references `axis_schema='exterior_polar_rho_u_v1'` and `_EXTERIOR_POLAR_AXIS_SCHEMA`. Librarian scope.
- **INS-17-001**: Test file `cogwheel/tests/test_lensing_exterior_carrier.py` still untracked.
