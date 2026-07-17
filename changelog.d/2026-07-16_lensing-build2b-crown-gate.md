---
date: 2026-07-16
---
### Fixed: microlensed relative-binning likelihood passes its crown gate (Build 2b)

`likelihood.LensedRelativeBinningLikelihood` now agrees with its brute-force
oracle through the same `LensedWaveformGenerator` across all crown-gate
configs, including `near-cusp`. The per-bin amplification kernel is now reduced
by dense sub-sampling and per-bin least squares instead of a two-edge secant,
removing a caustic-aliasing blow-up in `(h|h)` (see FINDINGS F006). The
performance gate was also corrected: the contraction is checked for
subdominance against the per-eval special-function cost
(`_amplification_coefficients`) and the public `lnlike` is checked faster than
`lnlike_bruteforce`, rather than against the coarse-grid strain call — which is
a co-cost of relative binning, not its competitor (see FINDINGS F007). The
unlensed-limit (`F→1`) floor was confirmed to be a template-construction
asymmetry rather than a normalization error and is gated test-side by a
zero-noise anchor. No public API changed.
