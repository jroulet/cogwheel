---
date: 2026-07-22
section: likelihood
---
### Build 8g — far-field tiling + eps registration gate (complete)

The full-box campaign exposed that far-field placement was
fixture-scale legacy (a single hard-coded box per image-count region;
1/1024 prior draws served, 942 out-of-box). Fixed: (1) mass-stratified
exterior tiling of the prior's shear-frame y-support replaces the
single box (`_mass_strata`/`_stratum_w_range`/`_farfield_tiles`,
stratum ratio `R = sqrt(f_hi/f_lo)`, tiles admitted only wholly
outside the `caustic_reach + eta_max` disk; beyond-w-cap strata and
`max_farfield_regions` cap truncation both recorded loudly);
`max_farfield_regions` now defaults to `None` (uncapped, commit
e91550e); (2) a held-out max-normalized envelope-eps registration
gate (`tube_eps_max = 5e-2`, `farfield_eps_max = 3e-3`) excludes
gated charts from the packed artifact, round-tripping the decision
through per-chart provenance on resume; (3) a runtime foot-of-normal
curvature-radius tube skip (`_min_curvature_radius`) records charts
whose `eta_max` exceeds half the band's minimum caustic curvature
radius rather than training them wrongly. SPEC.md TRAINING narrative
updated to match.
