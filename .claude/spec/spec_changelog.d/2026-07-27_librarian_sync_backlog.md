---
date: 2026-07-27
bump: patch
---

Doc-sync correction (Librarian post-commit pass over the 14-commit backlog
ending d0dc6da): two Conventions-section statements had drifted from the
code they describe.

- Lensing delay frame: the single authoritative frame-origin construction is
  `channels._frame_delays(source, matrix)` (returns `(images,
  absolute_delays, t_min)` together, added in 74c1d55 so partition builders
  need not re-solve the image quartic); `_frame_t_min` is now a thin
  accessor over it, not the authoritative site itself.
- Marginalized-path distance convention: the posterior's `d_luminosity`
  column is the PHYSICAL luminosity distance on both the `LensedIASPrior`
  and `LensedMarginalizedExtrinsicIASPrior` routes (4ffbde5), not the
  APPARENT distance requiring a post-analysis `sqrt(mu_macro)` rescale as
  previously stated. `cogwheel/lensing/waveform.py` now carries the single
  authoritative statement of this convention.
