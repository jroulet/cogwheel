---
date: 2026-07-20
bump: minor
---
Build 8a surrogate speed layer: new public module
`cogwheel/lensing/surrogate.py` (`LensAmplificationSurrogate`) — an
offline-trained tensor cubic-spline emulator of the SACR-C envelope
over (log w, gamma, y1_eig, y2_eig) with exact beta elimination, a
refusal-conservative domain gate, and npz/pickle serialization; a new
additive `ChangRefsdalChannels.geometry_partition` method; and an
`amplification_surrogate` kwarg on the lensed likelihoods (default
`None`, crown byte-identical). Also trues the sampling-layer row to
the Build 7b state (both-parity gamma range; four-refusal posterior
net; deltoid fold validity).
