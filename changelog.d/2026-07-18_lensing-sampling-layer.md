---
date: 2026-07-18
---
### Microlensed events are now sampleable end-to-end

New `LensedIASPrior` and `LensedPosterior` make the microlensed
relative-binning likelihood run under cogwheel's standard samplers.
The sampled lens coordinates follow the reduced parametrization: the
redshifted lens mass (log-uniform; the lens redshift folds in exactly,
since only the combination `M_L(1+z_L) f` enters the physics), the
reduced shear, and the source position as a shear-frame unit box scaled
so the engine's certified domain is respected by construction; kappa is
never sampled (exact mass-sheet degeneracy) and no orbital-phase fold
is assumed beyond the (2,2) mode. Named engine refusals become
`lnL = -inf` exactly at the posterior boundary — the engine's
certified-or-refuse contract is untouched — and the ratio-layer
fiducial cache is excluded from pickling so parallel sampler workers
rebuild deterministically. Certified by prior round-trip, Jacobian,
domain-safety, folding, mass-sheet-invariance, and refusal-net gates.
