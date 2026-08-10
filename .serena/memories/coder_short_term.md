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
(INCLUDED: NEW ENTRY BELOW)

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
