# Coder Short-Term Observations

- WP1 ppGO interior-cell extrapolation: added `_extrapolate_floor` (line ~875) and 5 `_EXTRAP_*` constants (line ~197) in `cogwheel/lensing/ppgo_map.py`. Modified `_measure_cell` to store `per_angle_data` list, run extrapolation fallback for interior cells (rho_center < 1.0), and relax the `floor > w_ceiling` guard for interior cells only. Exterior cells are byte-identical to HEAD.
