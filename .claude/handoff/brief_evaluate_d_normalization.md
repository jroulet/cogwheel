# Build Brief: Evaluate d-axis normalization before full training

## Mission

Before launching the expensive full-prior training, determine whether the
far-field chart's `d` axis (signed perpendicular distance from caustic)
should be normalized by the local curvature radius R_c. If so, implement it.

## The question

Currently `d` is in absolute source-plane units. The Airy fold transition
happens at `d ~ R_c` (curvature radius), which varies from 0.016 (gamma=0.03)
to 0.157 (gamma=0.28) for positive parity. A spline in absolute `d`:
- At low gamma: needs many nodes in a tiny range [0, 0.016]
- At high gamma: needs the same density but over [0, 0.157]
- Interpolating across gamma with fixed d-grid is wasteful

If `d` were `d/R_c`, the transition would always be at O(1) and the same
node count would work at all gammas.

## What to evaluate

1. Does the current architecture (absolute d, separate charts per gamma band)
   already handle this? Each gamma band has its own d-range tuned to that
   gamma's caustic scale — so maybe it's fine as-is?

2. Or does the gamma AXIS of the spline (which interpolates across the band)
   suffer because the d-scale changes within a band?

3. Measure: for a representative far-field chart spanning gamma=[0.3, 0.6],
   compare held-out eps with (a) raw d axis vs (b) d/R_c axis at the same
   node count. If (b) is significantly better, implement it.

4. If normalization helps: implement `d_normalized = d / R_c(gamma, theta)`
   in the far-field chart training and serving. The R_c is available from
   `geometry.caustic_curvature_radius` (shipped in build 1a).

## Constraints

- This is an EVALUATION, not a blind change.
- If the answer is "current architecture is fine", document why and proceed
  to training.
- If the answer is "normalize d", implement it THEN retrain.
- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
