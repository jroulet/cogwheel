---
date: 2026-07-30
bump: minor
---

### TubeChart records gain a `theta_to_s` arc-length axis map

`lens_amplification_surrogate` npz records for near-caustic tube charts now
carry an additional per-chart array, `chart{i}_theta_to_s`, of shape
`(2, N_map)` = `[theta_fine, s_fine]`.

The tube chart's fourth interpolation axis is now **arc length**
`s = ∫ caustic_speed dtheta` along the fold arc, rather than raw `theta`. The
stored map is the single authoritative representation of that coordinate: the
same table places the training nodes and converts a query `theta` to `s` at
serve time (one `np.interp`, so no quadrature enters the likelihood's hot
path). Membership tests and cusp-window exclusion remain in `theta` and are
unchanged.

Additive and backward-compatible on the reader side: `TubeChart.from_values`
takes the map as an optional keyword argument and falls back to the identity
map `s = theta - theta_lo`, under which splining in `s` is the previous
`theta` spline shifted by a constant.

No trained artifact exists yet, so nothing on disk needs migrating — the
window in which this is free closes when the first surrogate is trained.
