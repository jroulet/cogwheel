---
bump: patch
---

### Schwinger quad-double extension — mpmath path for w > 60

SPEC.md "Microlensing engine" row updated to describe the two-tier Schwinger
path: dd-quadrature path certified up to `w <= W_CEILING_SCHWINGER = 60`;
mpmath arbitrary-precision path (`_f_schwinger_mpmath`, `dps = 30 + ceil(w)`,
paired N/2N certification) for `60 < w <= W_CEILING_SCHWINGER_QD = 150`;
refusal above 150. F019 note extended to three distinct ceilings.
`_SADDLE_W_CEILING` raised from 58 to 148 in `surrogate_training.py` to
match the new QD engine ceiling.
