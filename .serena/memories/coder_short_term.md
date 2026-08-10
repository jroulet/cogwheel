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
(INCLUDED: NEW ENTRY BELOW)

(fold_carrier_chart build, 2026-08-10):
- Added rho_carrier: np.ndarray | None field to ExteriorPolarChart (after rho_log_axis, default None).
- from_values: rho_carrier kwarg validates len==n_rho + all finite; demodulates envelope BEFORE carrier_rate: demod_fold = exp(-1j * w_grid[:,None,None,None] * rho_carrier[None,None,:,None]) then original carrier_rate demod runs on the already-fold-demodulated envelope. Passes rho_carrier to _assemble.
- _assemble: rho_carrier kwarg with same validation, passed to cls() constructor.
- _evaluate_chart exterior branch: after carrier_rate remodulation, if chart.rho_carrier is not None: rho_c_interp = float(np.interp(rho, chart.rho_grid, chart.rho_carrier)) then result *= np.exp(1j * np.exp(log_w_query) * rho_c_interp). Uses RAW rho (before log transform), NOT the log-transformed coordinate.
- _chart_to_npz exterior branch: writes rho_carrier to arrays when not None (conditional, same pattern as theta_to_u).
- _chart_from_npz exterior branch: reads via data.get(prefix+'rho_carrier') (optional, None fallback for v3 artifacts lacking key). Passes to _assemble.
- Schema bump: _EXTERIOR_POLAR_AXIS_SCHEMA_V3 removed, _EXTERIOR_POLAR_AXIS_SCHEMA_V4 = 'exterior_polar_rho_log_carrier_v1' become sole member of _KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS. _build_provenance + _chart_to_npz meta updated to V4. _validate_exterior_polar_axis_schema docstring updated.
- Test files that import _EXTERIOR_POLAR_AXIS_SCHEMA_V3 will get ImportError — test_dev to fix.

(ghost_excluded_tiles build, 2026-08-10):

(ghost_excluded_tiles build, 2026-08-10):
- Added _exclude_ghost_dominated(gamma, center, half, gamma_band=None) -> bool in surrogate_training.py, mirroring _exclude_near_cusp's gamma-band probe pattern. Maps tile corners+centre from (gamma, rho, theta_c) to eigenframe source via _from_caustic_fixed, builds macro_matrix(gamma, beta=0, kappa=0), probes ghost_kernel(w=[10.0]) at each point. GhostDomainError -> pass (retainable). contrib.delay.imag < _GHOST_DECAY_IM_THRESHOLD -> exclude. Domain refusals treated conservatively as retainable.
- Modified _farfield_exterior_tiles: added gamma_band and ghost_drop_count params (both optional, backward-compatible). Calls _exclude_ghost_dominated after _exclude_near_cusp in tile loop, increments ghost_drop_count[0] on exclusion.
- Wired in _train_band_charts: ghost_drop_count=[0] defined at 'exterior' in regions block, passed to _farfield_exterior_tiles with gamma_band=band, accumulated into exterior_region_report['ghost_excluded_tiles'].
- Saddle pproad (parity==-1, uses _farfield_tiles not _farfield_exterior_tiles) is unaffected.
- Imports _GHOST_DECAY_IM_THRESHOLD inside _exclude_ghost_dominated via `from cogwheel.lensing.chang_refsdal import channels` (lazy, avoids circular import at module level).

(rho_log_axis build, 2026-08-10):
- Renamed _EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER -> _EXTERIOR_POLAR_AXIS_SCHEMA_V3 with value 'exterior_polar_rho_log_v3'. Old tag removed from _KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS (hard-refuse on pre-v3 artifacts).
- Added rho_log_axis: bool = False field to ExteriorPolarChart (after carrier_rate).
- from_values: rho_log_axis=True validates rho_grid[0] > 1.0, computes ur_grid = log(rho_grid - 1.0), replaces the 3rd spline axis with ur_grid. Passes rho_log_axis to _assemble.
- _assemble: new rho_log_axis param, passes through to cls(...).
- _evaluate_chart exterior branch: when chart.rho_log_axis, v1 = math.log(v1 - 1.0) replaces raw rho.
- _chart_to_npz: meta adds 'rho_log_axis', axis_schema uses _EXTERIOR_POLAR_AXIS_SCHEMA_V3.
- _chart_from_npz: reads rho_log_axis = meta.get('rho_log_axis', False) for backward compat, passes to _assemble.
- _build_provenance: axis_schema -> _EXTERIOR_POLAR_AXIS_SCHEMA_V3.
- LensAmplificationSurrogate.from_engine: new rho_log_axis param, passed through to ExteriorPolarChart.from_values.
- _build_farfield_chart: passes rho_log_axis=True to from_engine for both parity branches.
- Test files importing _EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER will get ImportError — test_dev to fix.
