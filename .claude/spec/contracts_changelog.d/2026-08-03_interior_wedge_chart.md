---
bump: patch
---

### Document InteriorWedgeChart record format in lens_amplification_surrogate

DATA_CONTRACTS.yaml `lens_amplification_surrogate` description now covers
`InteriorWedgeChart` record fields: `axis_schema='wedge_caustic_relative_v1'`,
axes `(log w, gamma, r, theta_wedge)`, `wedge_map` (_WedgeCausticMap),
`refused_points` shape and coordinate, optional `theta_to_s` reparametrisation,
and `INTERIOR_SACR_C` envelope label.
