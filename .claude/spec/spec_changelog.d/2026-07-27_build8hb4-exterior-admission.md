---
bump: minor
---
### Build 8h-b4 — per-column exterior admission, saddle additive axis, gamma=1 guard

The far-field tiler's exterior admission test is replaced: a per-theta_c-
column probe (`_InteriorAdmission.admits_exterior`) against the exact
nearest-caustic distance supersedes the old single scalar exclusion disk
(`caustic_reach + eta_max`), which conflated the astroid caustic's
directional cusp spike with a uniform radius and built zero exterior
tiles above gamma~0.85 despite ~98% of that region being genuinely
exterior. The saddle (parity -1) exterior arm of the caustic-fixed radial
coordinate (`_to_caustic_fixed`/`_from_caustic_fixed`) switches from
multiplicative reach-normalisation to an additive scalar-reach offset,
since a directional caustic radius is ill-posed for the saddle's two
disjoint deltoid lobes; the interior and astroid-exterior arms are
unchanged. `_box_region_labels` now catches the named refusal errors
around the box-centre coordinate map and returns `(None, None)` instead
of crashing when a chart's box centre sits exactly on the `gamma = 1`
parity wall.
