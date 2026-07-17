---
date: 2026-07-16
---
### Added: microlensed waveform generator and relative-binning likelihood (Build 2)

Two new public entry points under `cogwheel/lensing/`, built on the
`chang_refsdal` engine: `waveform.LensedWaveformGenerator` applies the
Chang--Refsdal wave-optics amplification `F(w(f))` to a wrapped
`WaveformGenerator` per harmonic mode (`w = 8*pi*G*M_L*(1+z_L)*f/c**3`, linear
in `f`), and `likelihood.LensedRelativeBinningLikelihood` evaluates a fast
relative-binning log-likelihood for microlensed CBC signals. Both are additive
(no existing API changed) and support positive-parity macro images only —
configurations with `1 - kappa <= |gamma|` raise `geometry.LensDomainError`
rather than returning a degraded result.
