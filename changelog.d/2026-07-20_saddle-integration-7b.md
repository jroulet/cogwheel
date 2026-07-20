---
date: 2026-07-20
---
### Both parities, end to end

Macro-saddle lens hosts are now sampleable through the full pipeline:
the channel layer serves them by delegating to the engine's own
parity dispatch, the lensed waveform generator constructs on them, and
the sampled reduced-shear prior spans both parities in one continuous
range — the parity boundary is a measure-zero named refusal, so the
posterior carries no artificial cut between minima and saddles.  The
new channel-layer certification pins the saddle path to an independent
high-precision oracle at the 1e-9 gate (measured 5e-15), bounds the
switched kernels on and off the caustic, checks kernel continuity
across the two deltoid lobes, and reproduces the flat macro-
magnification plateau at the channel layer.  Test authorship also
falsified a planning-stage hypothesis: the residual strong-shear
likelihood gap against brute force is a relative-binning/noise effect,
not an envelope-resolution one, and is gated accordingly.
