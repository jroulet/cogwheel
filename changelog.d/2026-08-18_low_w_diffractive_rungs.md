---
date: 2026-08-18
---

Lensing: the microlensing wave-optics engine now serves the band bottom
(low `w`, near-diffractive regime) analytically on both parities via a
new module, `cogwheel/lensing/chang_refsdal/_diffractive.py`. Positive
parity serves a closed-form reduced-shear expansion admitted by a
truncation certificate; the macro saddle's low-w series diverges at
every order, so its band bottom is instead hosted by the exact 1D
Schwinger-parameter quadrature under the paired N/2N certificate. On a
10k-draw engine-free demand census, the new rungs serve 14.27% of the
prior analytically and host a further 14.93% on the exact engine under
certificate, dropping total exact-engine demand (`engine_residual`)
53.30% -> 24.10%. Program-to-date: `engine_residual` fell from 72.25% to
24.10% across the c3 band-split and low-w diffractive builds, in one day
of purely analytic-rung work with zero surrogate charts trained.
