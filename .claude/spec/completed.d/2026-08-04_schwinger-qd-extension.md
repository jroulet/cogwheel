---
date: 2026-08-04
section: Backlog
---

### Schwinger quad-double extension (mpmath path for w > 60)

Extended the Schwinger t-integral core beyond the original dd-quadrature
ceiling of `w = 60` using an mpmath arbitrary-precision path.

**WP1:** `_f_schwinger_mpmath()` added to `cogwheel/lensing/chang_refsdal/_schwinger.py`.
Lazy mpmath import (no overhead on the dd path), `dps = 30 + ceil(w)`,
paired N/2N node-doubling certification. New public constant
`W_CEILING_SCHWINGER_QD = 150`.

**WP2:** `cogwheel/lensing/chang_refsdal/operator.py` routes nodes with
`60 < w <= 150` through sequential mpmath evals. `_SADDLE_W_CEILING` in
`surrogate_training.py` raised from 58 to 148 to match the new ceiling.

Certified by `cogwheel/tests/test_lensing_schwinger.py` (extensive mpmath
oracle tests across the dd and QD bands; refusal at w=151).

SPEC.md updated: Schwinger exact-wave passage revised to describe the
two-tier path and extended F019 note (three distinct ceilings).
