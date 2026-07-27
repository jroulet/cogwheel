---
date: 2026-07-27
---

### Fixed: lensed-distance convention documented inconsistently

`cogwheel.lensing.marginalized_likelihood` and
`LensedMarginalizedExtrinsicIASPrior` described the `d_luminosity` column as the
apparent distance, requiring `d_L = d_app * sqrt(mu_macro)` in post-analysis.
`LensedIASPrior` described the opposite. The column is physical on both routes,
so the rescaling instruction would have inflated reported distances by
`sqrt(mu_macro)` (2.3x at `gamma = 0.9, kappa = 0`, growing as `gamma -> 1`).

`F` multiplies the strain with no compensating normalisation and carries
`F(w -> 0) = sqrt(mu_macro)`, so distance enters the amplitude as the physical
value by two distinct routes: `LensedIASPrior` transforms sampled `d_hat` to
standard `d_luminosity` via `UniformLuminosityVolumePrior`; the marginalized
prior has no distance parameter at all and its blob column comes from
`CoherentScoreHM._sample_distance(d_h, h_h)`, where `_get_dh_hh_timeshift` has
folded `F` into `d_h` and `|F|**2` into `h_h`.

No computed result was affected -- nothing in code applied `sqrt(mu_macro)` to a
distance. The convention now has one authoritative statement, in
`cogwheel/lensing/waveform.py` where `F` meets the strain, naming both routes;
the prior and likelihood docstrings defer to it.
