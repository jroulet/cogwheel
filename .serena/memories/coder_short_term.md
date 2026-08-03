# Coder Short-Term Observations

- INS-2-001 fix: `_subdivide_farfield_tile` (line ~3655) in `cogwheel/lensing/surrogate_training.py` had same stale ternary eff_w_nodes pattern as the main tiler. Applied identical 3-way if/elif/else: tile override -> interior/lobe_interior uses config.interior_w_nodes_per_decade -> else config.w_nodes_per_decade. Mirrors the pattern at line ~4327 in _train_band_charts.


- WP1 interior_w_nodes_per_decade: added `interior_w_nodes_per_decade: int = 15` field (line ~269) to `TrainingConfig` in `cogwheel/lensing/surrogate_training.py`. Modified `_train_band_charts` eff_w_nodes logic (line ~4324-4329) from a ternary to a 3-way if/elif/else: tile override -> interior/lobe_interior uses config.interior_w_nodes_per_decade -> else config.w_nodes_per_decade. Changed both `build_lobe` and `build_ff` closures' default-arg binding from `w_nodes=tile_w_nodes` to `w_nodes=eff_w_nodes` so the resolved integer (not None) is passed to the chart builders. Report `n_w_per_decade` was already reading `eff_w_nodes` — no change needed there.


- WP1 ppGO interior-cell extrapolation: added `_extrapolate_floor` (line ~875) and 5 `_EXTRAP_*` constants (line ~197) in `cogwheel/lensing/ppgo_map.py`. Modified `_measure_cell` to store `per_angle_data` list, run extrapolation fallback for interior cells (rho_center < 1.0), and relax the `floor > w_ceiling` guard for interior cells only. Exterior cells are byte-identical to HEAD.
