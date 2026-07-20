---
date: 2026-07-19
bump: minor
---
Build 7a engine hardening: runtime index-theorem census guard (F012
dead zone now a named refusal, both parities, with Morse-theory-correct
degenerate pass-throughs for fold-merged and on-cusp censuses),
cross-parity strong-shear Schwinger fallback in `F_op`/`F_op_grid`
(positive-parity `CancellationError` refusals at w <= 60 and gamma' > 0
become certified answers; certified outputs byte-frozen; shear-free and
above-ceiling refusals stand), an exactly-singular-Hessian named
refusal in the stationary-phase kernel replacing a raw `LinAlgError`
crash surfaced in production (F015), and the `LensedPosterior` refusal
net extended to the full named vocabulary (adds
`SchwingerCertificationError`, `LensedBinningError`).
