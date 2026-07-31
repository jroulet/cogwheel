---
date: 2026-07-31
bump: minor
---

### Far-field surrogate charts use gamma-resolved fold coordinates

`lens_amplification_surrogate` FarFieldChart records now persist the required
gamma-resolved `arc_map` and interpolate their spatial axes in fold-adapted
`(s, d)`: caustic arc length and signed nearest-fold distance. Refusal points
and spacing are stored in that same coordinate system.

This replaces the retired caustic-fixed `(rho, theta_c)` axes with no legacy
reader mode. Every FarFieldChart record carries the required
`axis_schema='farfield_arclength_s_perp_d_framewinv'`; loading validates that
tag and hard-refuses absent or unknown schemas rather than serving a stale or
wrong-frame artifact. Macro-saddle far-field charts intentionally remain
unavailable: those queries fall through to the exact engine until a
per-deltoid-edge design is certified.
