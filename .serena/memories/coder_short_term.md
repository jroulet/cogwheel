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
- (carrier_demod_1 build, 2026-08-10):
- Added carrier_rate field (float, default 0.0) to ExteriorPolarChart.
- from_values: when carrier_rate != 0, demodulates envelope by exp(-1j * carrier_rate * w_grid[:,None,None,None]) before spline fitting.
- _evaluate_chart ExteriorPolarChart branch: re-modulates via exp(1j * carrier_rate * w_query) after spline when carrier_rate != 0.
- LensAmplificationSurrogate.from_engine: after carrier continuity check (unchanged, on pre-demod envelope), estimates per-node k_node via unwrap+finite-diff, median → k_chart, demodulates envelope arrays, passes carrier_rate=k_chart to from_values.
- _chart_to_npz exterior polar branch: writes carrier_rate in meta, axis_schema → new tag _EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER = 'exterior_polar_carrier_demod_v2'.
- _chart_from_npz exterior polar branch: reads carrier_rate via meta.get('carrier_rate', 0.0) for backward compat, passes to _assemble.
- _assemble: accepts carrier_rate param with validation (must be finite).
- _build_provenance: uses _EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER.
- _KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS = frozenset({_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER}); old 'exterior_polar_rho_u_v1' tag removed → hard-refuses at load.
- Test files that import _EXTERIOR_POLAR_AXIS_SCHEMA directly will get ImportError (renamed to _EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER) — test_dev to fix.
