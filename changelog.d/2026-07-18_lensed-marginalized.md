---
date: 2026-07-18
---
### Microlensed posterior sampling in a 13-dimensional space

`LensedMarginalizedExtrinsicLikelihood` marginalizes sky location,
coalescence time, polarization, orbital phase, and amplitude for the
microlensed model by reusing the coherent-score (higher-mode) machinery
unchanged: because the dimensionless lens frequency is linear in `f`,
each image's delay phase is an exact matched-filter time shift, so the
lensed timeseries is the image-summed, kernel-weighted contraction
against the same fiducial summary weights as the unlensed case, and the
norm is the `|F|^2`-weighted bin sum under the existing lens-aware bin
guard. Engine refusals resolve before the marginalization integral is
ever attempted. The sampled space is 12 intrinsic + 7 lens standard
parameters (extrinsics drawn from their conditionals in
postprocessing), and the posterior's distance column is the apparent
distance `d_app = d_L/sqrt(mu_macro)` with the physical transform a
documented post-analysis step. Paired registered prior:
`LensedMarginalizedExtrinsicIASPrior`.
