# Coder Short-Term Observations

(exterior_cusp_exclusion build, 2026-08-09):
- Added _deltoid_cusp_source_angles(gamma, n) -> list[float] for saddle/deltoid
  cusp angles D₂-folded into [0, π/2], mirroring _lobe_cusp_source_angles but
  measuring from origin (not lobe centroid). Sweeps both branches around both
  lens-plane centers 0 and π.
- Modified _exclude_near_cusp: added optional gamma_band parameter. When
  provided, checks at gamma_lo, gamma_mid, gamma_hi. Backward-compatible
  (gamma_band=None falls back to single-gamma check).
- Modified _farfield_tiles: added keyword-only args cusp_angles, gamma,
  gamma_band. When provided, calls _exclude_near_cusp and continues (skips
  tile). Backward-compatible.
- Wired saddle exterior: _train_band_charts now computes deltoid cusp angles
  via _deltoid_cusp_source_angles in the parity != 1 branch and passes them
  to _farfield_tiles with gamma=gamma_mid, gamma_band=band.
- Bumped _CUSP_EXCLUSION_DISTANCE from 0.2 to 0.35. Updated docstring to
  cover both astroid and deltoid cusps. Removed "astroid only" claim.
- Wrote scripts/measure_cusp_exclusion.py probe to calibrate the constant.
- Lobe interior unchanged: _SaddleLobeAdmission.admits comment confirms
  _LOBE_CUSP_EXCLUSION_DISTANCE is retired.
- Orphaned comment line "#: Deltoid-lobe interior near-cusp carve-out
  distance..." on former _LOBE_CUSP_EXCLUSION_DISTANCE line left as-is
  (cosmetic only, not in scope).
- ((carrier_demod_1 build, 2026-08-10):...
- Test files that import _EXTERIOR_POLAR_AXIS_SCHEMA directly will get ImportError (renamed to _EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER) — test_dev to fix.
(fold_carrier_training build, 2026-08-10):
- Added _needs_fold_carrier(gamma, center, half, gamma_band=None) -> bool in surrogate_training.py, mirroring _exclude_ghost_dominated's coordinate-mapping pattern but returning True iff ghost_kernel SUCCEEDS (ghost exists) at ANY probed point, regardless of Im(tau_c).
- Removed _exclude_ghost_dominated check from _farfield_exterior_tiles: ghost-dominated tiles now flow through to training (no longer dropped). ghost_drop_count parameter retained but stays 0.
- Added 'fold_carrier': _needs_fold_carrier(...) key to exterior tile dicts in _train_band_charts.
- Added fold_carrier: bool = False kwarg to _build_farfield_chart, threaded through to LensAmplificationSurrogate.from_engine.
- build_ff closure reads fold_carrier from tile dict and passes to _build_farfield_chart.
- Added fold_carrier: bool = False kwarg to from_engine in surrogate.py.
- Added _compute_rho_carrier(gamma_grid, rho_grid, theta_c_grid, w_grid) -> np.ndarray|None: probes ghost_kernel at every (gamma, theta_c) node for each rho, takes median(Re(tau_c)) over valid nodes; returns None if no valid nodes.
- When fold_carrier=True in from_engine: computes rho_carrier, temporarily demodulates envelope by exp(-1j*w*rho_carrier[None,:]) for continuity check + k_chart estimation, passes rho_carrier to from_values. When fold_carrier=False: byte-identical to HEAD.
- Subdivision (3334) and reprovision (4346) call sites of _build_farfield_chart use default fold_carrier=False (backward-compatible).
(foreman-fix build, 2026-08-10):
- INS-1-001: Changed np.exp(log_w_query) -> np.exp(log_w_clamped) in _evaluate_chart fold-carrier re-modulation (surrogate.py:2894). The carrier_rate re-modulation uses log_w_clamped; fold-carrier phase must match to preserve phase cancellation on low-w extrapolation queries.
- INS-1-002: test_ghost_excluded_tiles_is_positive_integer renamed to test_ghost_excluded_tiles_is_zero, assertion changed from ghost_ct > 0 to ghost_ct == 0 (ghost-dominated tiles now rescued by fold-carrier, not dropped). Self-falsification test updated to match.
- INS-1-003: All 6 occurrences of 'exterior_polar_rho_log_v3' in test_lensing_surrogate.py replaced with 'exterior_polar_rho_log_carrier_v1'. Added test_rho_log_v3_schema_raises_valueerror to ExteriorPolarStaleSchemaHardRefusalTestCase.
(exterior_2d_fold_carrier build, 2026-08-10):
- Renamed rho_carrier -> rho_u_carrier on ExteriorPolarChart (field shape (n_rho, n_theta_c) instead of 1D (n_rho,)).
- Added _EXTERIOR_POLAR_AXIS_SCHEMA_V5 = 'exterior_polar_rho_u_carrier_v2'; V4+V5 in _KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS.
- Extracted _probe_ghost_delay(gamma, rho, theta_c, matrix, w0) -> float|None helper.
- Replaced _compute_rho_carrier with _compute_rho_u_carrier: 2D output, median over gamma, NaN fill along u then rho, all-NaN -> None.
- from_values: 2D shape validation, demodulates with rho_u_carrier[None,None,:,:] (2D broadcast).
- _assemble: 2D shape validation.
- _evaluate_chart exterior branch: bilinear (rho, u) interpolation via searchsorted on u_axis (built from theta_c_grid via theta_to_u if present), lerp along u, np.interp along rho.
- from_engine: calls _compute_rho_u_carrier; 2D broadcast rho_u_carrier[None,None,:,:] for continuity gate + k_chart estimation.
- _chart_to_npz: writes 'chart{i}_rho_u_carrier' key (2D), meta uses V5.
- _chart_from_npz: tries 'rho_u_carrier' first; falls back to 'rho_carrier' with np.broadcast_to(rho_1d[:,None], (n_rho, n_theta_c)) for old 1D artifacts.
- _build_provenance: V5 tag.
- _validate_exterior_polar_axis_schema docstring updated to list V4+V5.
- All 52 'rho_carrier' references renamed to 'rho_u_carrier' except backward-compat NPZ key 'rho_carrier'.
- _compute_rho_carrier deleted. No 'rho_carrier' field on ExteriorPolarChart. surrogate_training.py zero changes.
- Test files will need updates (rho_carrier -> rho_u_carrier, 1D -> 2D shape assertions).
(INS-1-001 fix, 2026-08-10):
- Updated test_lensing_surrogate_training.py: 6 occurrences of '_compute_rho_carrier' -> '_compute_rho_u_carrier' (2 mock strings + 4 comment references).
- zero_carrier shape: np.zeros(n_rho) -> np.zeros((n_rho, _DEFAULT_PARAM_NODES)) for 2D carrier.
- Added _DEFAULT_PARAM_NODES to imports from cogwheel.lensing.surrogate (removed again in INS-1-001 v2 fix — replaced with literal 4).