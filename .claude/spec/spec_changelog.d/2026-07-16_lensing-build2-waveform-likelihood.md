---
bump: minor
---

### Microlensed waveform generator + relative-binning likelihood (Build 2)

New architecture-layer row for the two Build-2 production modules that sit on
the completed `chang_refsdal` engine. `cogwheel/lensing/waveform.py` adds
`LensedWaveformGenerator`, which composes an ordinary `WaveformGenerator` and
multiplies every harmonic mode by the shared Chang--Refsdal factor `F(w(f))`
(`w = 8*pi*G*M_L*(1+z_L)*f/c**3`, dimensionless and linear in `f`), exposing
the per-image `(tau_a, K_a)` decomposition alongside the collapsed total.
`cogwheel/lensing/likelihood.py` adds `LensedRelativeBinningLikelihood`, a
`BaseLinearFree` subclass that heterodynes against an unlensed reference and
reconstructs the lensed `(d|h)`/`(h|h)` from delay-continuous frequency-moment
summaries contracted mode-then-image (additive `M^2 + n_img^2`, FFT-free hot
path), with analytic image-delay phases, interpolated kernels, and a lens-aware
bin guard (`LensedBinningError`). Positive-parity macro images only
(`1-kappa > |gamma|`, enforced by raising `geometry.LensDomainError` at the API
boundary). Both are in-memory `JSONMixin` objects — no new on-disk data
product. Build 3 (sampled lens coordinates, astroid folding,
injection-recovery validation) remains pending.
