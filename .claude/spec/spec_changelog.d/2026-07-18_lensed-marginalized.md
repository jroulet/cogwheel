---
date: 2026-07-18
bump: minor
---
Marginalized lensed likelihood (Build 5): `LensedMarginalizedExtrinsicLikelihood`
+ registered `LensedMarginalizedExtrinsicIASPrior` — coherent-score
(higher-mode) extrinsic marginalization for the microlensed model via
exact per-image time shifts through the unchanged fiducial weights and
`|F|^2`-scaled norms; refusals precede the integral; drawn distance is
apparent distance d_app (F009 transform deferred to post-analysis).
New test module `cogwheel/tests/test_lensing_marginalized_likelihood.py`.
