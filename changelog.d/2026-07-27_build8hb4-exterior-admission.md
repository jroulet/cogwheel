---
date: 2026-07-27
---
### Per-column exterior admission, saddle additive axis, gamma=1 guard (Build 8h-b4)

The far-field trainer's exterior tile admission now probes each tile's
theta_c column against the exact nearest-caustic distance instead of
testing against a single scalar exclusion disk, fixing a coverage
collapse above gamma~0.85 where the disk (built from the astroid's
directional cusp spike) exceeded the entire prior source box and
admitted zero exterior tiles. The saddle-parity exterior arm of the
caustic-fixed radial coordinate switches from multiplicative
reach-normalisation to an additive scalar-reach offset (a directional
caustic radius is undefined for the saddle's two disjoint deltoid
lobes); the interior and astroid-exterior arms are unchanged. Chart
construction no longer crashes when a box centre lands exactly on the
`gamma = 1` parity wall — it now records an unknown image_count/parity
for that chart instead.
