# Coder Short-Term Observations

- WP1 edge-coincidence tolerance in `_lobe_cusp_axis_map` (surrogate.py,
  2026-08-15): relaxed the two strict cusp-vs-edge guards (`if not
  cusp_angle > theta_hi: raise` / `< theta_lo`) to admit a cusp coincident
  with the side-appropriate edge within `_CUSP_EDGE_COINCIDENCE_ULPS = 8`
  ULPs (tol = ULPS*eps*max(1,|edge|,|cusp|)); d at that edge clamped to 0
  via `max(..., 0.0)`; keep-map semantics (return type stays non-Optional
  tuple) chosen over None to avoid the `_chart_from_npz` unconditional
  `theta_to_u` read -> KeyError trap the Professor flagged. Genuine
  interior straddle STILL raises ValueError ('...a straddle.'). Const name
  ends `_ULPS`, NOT in the part0 absorber regex suffix list
  (_EPS|_MARGIN|_FRAC|_STANDOFF|_SAFETY), so no allowlist needed. Smoke-
  tested engine-free: logged pair (0.0, 3.552713678800501e-16,
  3.270275691376951e-16,'right') now returns monotone map, tf[0]==0.0,
  tf[-1]==3.55e-16, uf[0]==0.0; interior straddle raises both sides;
  outside-edge regression unchanged.
- SIBLING AUDIT (grep `requires cusp_angle` in surrogate.py): ONLY
  `_lobe_cusp_axis_map` (lines 671/682 pre-edit) had the strict-inequality-
  at-coincidence shape. `_wedge_cusp_axis_map` takes `origin` ('low'/'high')
  and pins its cusp to the domain boundary (theta=0 / pi/2) — no cusp-vs-edge
  guard, safe by construction. `_deltoid_cusp_axis_map` uses non-strict
  `if cusp_angle <= theta_lo: ... else:` branch selection + straddle->None
  (`if theta_lo < cusp_angle < theta_hi: return None`) — safe. Neither
  needed a fix.
