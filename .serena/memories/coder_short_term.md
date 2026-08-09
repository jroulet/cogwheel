# Coder Short-Term Observations

WP1 ExteriorPolarChart cusp-adapted u coordinate (2026-08-08):
- Added theta_to_u: np.ndarray | None field to ExteriorPolarChart
- Mirrored lobe/wedge pattern: from_values/from_engine/_assemble all accept
  optional theta_to_u/u_grid; when provided, spline is fit on uniform u_grid
- _evaluate_chart remaps theta_c->u via np.interp before spline contraction
- _chart_to_npz conditionally writes, _chart_from_npz reads via data.get() (returns None on missing key, matching chart's own type annotation)
- Bumped axis schema: 'exterior_polar_rho_theta_c' -> 'exterior_polar_rho_u_v1'
- _build_farfield_chart trains parity==1 tiles with cusp-adapted u via _wedge_cusp_axis_map;
  saddle exterior (parity==-1) passes None (raw-theta fallback)
- Added _uniform_axis to surrogate_training.py imports
- FIX INS-1-001: changed data[prefix+'theta_to_u'] to data.get(prefix+'theta_to_u') in all three branches of _chart_from_npz (lobe, wedge, exterior-polar) so NPZ round-trip works for theta_to_u=None charts
- FIX INS-2-001: updated test_new_schema_without_theta_to_u_raises_keyerror -> test_new_schema_without_theta_to_u_loads_with_none: now asserts chart loads with theta_to_u=None instead of expecting KeyError