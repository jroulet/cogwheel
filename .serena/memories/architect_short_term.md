# Architect Short-Term Observations

Build brief_ppgo_interior_certification: (previous build, retained for context).

Build brief_fold_corrected_ppgo: (previous build, retained for context).

Build brief_interior_wedge_chart: (previous build, retained for context).

Build brief_wedge_followup: Two fixes to `from_wedge_engine` in surrogate.py. (1) r-dependent w-ceiling: the brief's formula uses r_min but Professor+Simplifier confirm it should be r_max (r_grid[-1]) for zero refusals — `w_max = min(w_range[1], 58.0 / (r_grid[-1] * reach_max))`. Define local `_DD_PRODUCT_MARGIN = 58.0` since surrogate.py cannot import from surrogate_training.py. (2) Arc-length axis: compute at representative gamma (median of gamma_grid) via geometry.caustic_speed + cumulative_trapezoid; span the tile's theta_wedge_range (validator requires theta_fine[0] == theta_wedge_grid[0]); store as (2, N_map) theta_to_s, compute s_grid by interpolating theta nodes through the map, pass both to from_wedge_values. All serve-time plumbing already exists. Single WP — both features are ~20 lines combined in one function.