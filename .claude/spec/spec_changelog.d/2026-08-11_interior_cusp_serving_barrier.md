---
date: 2026-08-11
bump: patch
---
### Pearcey arm: interior cusp serving + ppGO fold-band gate

Updated the `_pearcey_cusp.py` description in the Microlensing engine row
(Key abstractions): the high-w ppGO fast rung now additionally requires the
source OUTSIDE the fold arm's serving band (`nearest.distance >= _ETA_MAX_FOLD`
— inside the band the fold arm is the designated rung), and the paragraph
gains an INTERIOR CUSP SERVING sentence: interior sources (3 real stationary
points, `rho < 1`) bypass the per-image delay calibration certificate — the
uniform-error gate `radius >= radius_min` bounds the answer to the envelope
bar and the ratio `P/P_asymp` is self-calibrating to leading order — while
exterior sources (1 stationary point) still validate delay-to-image
alignment.
