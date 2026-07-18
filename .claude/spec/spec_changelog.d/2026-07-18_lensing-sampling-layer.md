---
date: 2026-07-18
bump: minor
---
Microlensed sampling layer (Build 4): `LensedIASPrior` (sampled reduced
lens coordinates — redshifted ln lens mass, reduced shear, shear-frame
source box; kappa/beta/z_lens eliminated; astroid quadrant folding; no
phase-fold) and `LensedPosterior` (named engine refusals mapped to
lnL = -inf at the posterior boundary only); fiducial cache dropped on
pickle for fork-safe determinism. New row in the Layers table; new test
module `cogwheel/tests/test_lensing_prior.py`.
