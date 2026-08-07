# Coder Short-Term Observations

Added `regions` filter to training entry points (WP1):
- `_train_band_charts` signature: added `regions: tuple[str, ...] | None = None`; `None` → `('tube', 'exterior', 'wedge_interior', 'lobe_interior')`
- Tube section: `if 'tube' in regions:` wraps `tube_w_range` computation; for-loop iterates empty when tube not in regions
- Exterior section: `if 'exterior' in regions:` wraps exterior tiles, ppgo boundary, and region report; `else` sets defaults (`exterior_tiles=None`, `region_exclusion_rho=exclusion_rho`)
- Lobe section: `if 'lobe_interior' in regions:` wraps the body of `if parity != 1:`
- Wedge section: `if 'wedge_interior' in regions:` wraps the body of `else:` (wedge branch)
- Dispatch loop at bottom is unguarded (iterates `admitted`, which is empty for skipped regions)
- `exterior_admission = None` initialized before exterior guard (avoid NameError in dispatch loop)
- `train()`: added `regions` kwarg, threads to `_train_band_charts`
- `scripts/train_lens_surrogate.py`: added `--regions` with `nargs='*'`, `choices=[...]`, converts to tuple and threads to `train()`

Added ExteriorPolarChart (WP1 exterior_polar_rechart):
- ExteriorPolarChart: frozen dataclass with (rho_grid, theta_c_grid) axes, NO arc_map
- _exterior_polar_serves gate: np.isfinite guard, box containment, exclusion balls, image_count, eta
- from_engine: polar (rho_range, theta_c_range) API, no interior path
- select_chart: tube > exterior_polar > lobe > wedge
- _chart_to_npz / _chart_from_npz: kind='exterior_polar'
- surrogate_training.py, surrogate_census.py: imports and references updated
- Added ExteriorPolarChart training-pipeline rewire (WP2): deleted _farfield_box_to_smooth bridge + _saddle_arc_branch; removed parity!=1 refusal from _build_farfield_chart; added _CUSP_EXCLUSION_DISTANCE + _exclude_near_cusp; cusp carve-out in _farfield_exterior_tiles via new `gamma` kwarg; _load_or_build catches schema mismatch; surrogate.py comment updated. Saddle axis edges NOT aligned (deltoid is off-axis). OWED: full test suite update for polar API; delete (s,d) symbols from surrogate.py; update 3 test files referencing _farfield_box_to_smooth. (s,d) symbols from surrogate.py
