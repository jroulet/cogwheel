---
date: 2026-07-30
bump: minor
---

### Tube charts interpolate in arc length, not raw theta

The surrogate row's near-caustic TUBE chart coordinates now read
`(gamma, u = sqrt(eta), s, log w)`, where `s = ∫ caustic_speed dtheta` is arc
length along the fold arc. The chart interpolates in `s`; a query `theta` is
converted through the per-chart `theta_to_s` map stored with the chart.
Membership tests and cusp-window exclusion remain in `theta`.

Splining in the raw angle made held-out eps depend on where the arc bounds
happened to fall — measured `±23%` swing under a `±0.01 rad` bound shift
(F042). Arc length is insensitive to that shift and reaches a lower eps at
every node count tested.

This is the spec surface the 1e-tube Inspector flagged as `INS-1-001` in all
three of its revision rounds. No agent in that loop was scoped to `SPEC.md`,
so the finding could not be resolved from inside the build; recorded as such
in FINDINGS, since the same shape exhausted the revision budget in two earlier
builds.
