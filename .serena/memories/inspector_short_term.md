## 2026-08-10 — Review: ghost-gate tile exclusion (brief_exterior_rho_phase_carrier, WP-1)

Reviewed the uncommitted diff on `claude-dev`. Build adds `_exclude_ghost_dominated()` in surrogate_training.py, wired into `_farfield_exterior_tiles` for positive-parity tiles.

### Verified correct
- `_exclude_ghost_dominated` probes tile corners + centre via `geometry.ghost_kernel(w=[10.0], ...)`, checking `Im(tau_c) < channels._GHOST_DECAY_IM_THRESHOLD` (=0.4)
- Conservative fail-safe: `GhostDomainError`, `LensDomainError`, `ValueError` all fall through to retain (engine serves it)
- `gamma_band` probes gamma_lo, gamma_mid, gamma_hi — matches `_exclude_near_cusp` pattern
- `ghost_drop_count: list[int] | None = None` mutable-counter pattern correct (default None, guarded access)
- Wired to `exterior_region_report['ghost_excluded_tiles']` — report key additive only
- Backward compatible: `gamma`/`gamma_band`/`ghost_drop_count` all optional with None defaults; no existing callers broken
- All 86 exterior admission tests pass, all 88 surrogate training tests pass, 110 surrogate tests pass — no regressions
- 7 self-falsification test classes prove detectors are not vacuous
- Import chain clean (delayed `from cogwheel.lensing.chang_refsdal import channels` inside function body)
- No spec/DATA_CONTRACTS divergence introduced (additive mechanism only)

### Findings
None. Implementation is correct and complete for WP-1.

### STILL OPEN (carried from previous reviews, not induced by this change)
- **INS-16-002**: SPEC.md line 63 still says `'exterior_polar_rho_u_v1'` — should be `'exterior_polar_carrier_demod_v2'`. Librarian scope.
- **INS-16-003**: DATA_CONTRACTS.yaml line 199 still references `axis_schema='exterior_polar_rho_u_v1'`. Librarian scope.
- **INS-17-001**: Test file `cogwheel/tests/test_lensing_exterior_carrier.py` still untracked.
- **INS-17-002**: `n_w = log_w_grid.size` at surrogate.py (dead code). [trivial]
