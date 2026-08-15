# Architect Short-Term Observations

2026-08-15 lobe_cusp_axis_edge_tolerance: 7a smoke crash in _lobe_cusp_axis_map
(surrogate.py:682). Root: `_lobe_nearest_cusp` (surrogate_training.py:5581) picks
`side` by comparing cusp to tile CENTER, but `_lobe_cusp_axis_map` guards cusp
vs tile EDGE (strict cusp>theta_hi / cusp<theta_lo). A tile whose center is left
of a cusp but whose upper edge reaches/crosses it = machine-precision STRADDLE
(cusp 3.27e-16 vs theta_hi 3.55e-16, 2.8e-17 inside) -> strict guard raises.
Sibling audit: _wedge_cusp_axis_map cusp fixed at 0/pi-2 boundary, no edge guard
-> different shape, safe. _deltoid_cusp_axis_map ALREADY handles it: straddle->None
+ non-strict <= branch; returns Optional, callers None-handle. _lobe_cusp_axis_map
returns non-Optional tuple; 3 callers don't handle None. Options: (a) tolerance-
relax edge guards + clamp d->0 at coincidence (local) vs (b) mirror _deltoid
Optional straddle->None. Pending professor/simplifier ruling.
