2026-08-06 session (2nd item): fixed INS-1-001, a stale docstring sentence in
_build_wedge_chart (surrogate_training.py ~L2991-2992) that claimed
from_wedge_engine "builds the caustic arc-length theta_wedge -> s map
INTERNALLY" -- that arc-length map was retired for the wedge path in favor of
the cusp-adapted angular (u = d**(2/3)) map built via _wedge_cusp_axis_map.
Reworded to "builds the cusp-adapted angular (u = d**(2/3)) theta_wedge -> u
map INTERNALLY", now consistent with the docstring's own later
axis_origin/"cusp-adapted angular map" sentence. Single replace_content call,
verified via ast.parse. No code (only docstring) changed -- matches the
finding's note that "Code is correct; only the docstring is stale."