## 2026-08-04

### Schwinger quad-double extension — mpmath path for 60 < w ≤ 150

The exact Schwinger wave evaluator gains a second tier for
`60 < w <= W_CEILING_SCHWINGER_QD = 150`:

- **`_schwinger._f_schwinger_mpmath`**: lazy-imported `mpmath` evaluated at
  `dps = 30 + ceil(w)`, with paired N/2N certification at the same 3e-10
  tolerance as the dd path.
- **`operator.py`**: the per-node routing in both `_positive_parity_grid`
  and `_saddle_grid` now forwards `w ∈ (60, 150]` to sequential mpmath
  evaluations rather than issuing a named refusal.
- **`surrogate_training.py`**: `_SADDLE_W_CEILING` raised from 58 to 148
  to match the new engine ceiling.
- Above `w = 150` the evaluator still refuses by name
  (`SchwingerCertificationError`).

There are now **three distinct ceilings** (F019 — extended):
`W_CEILING_SCHWINGER = 60` (dd-quadrature frequency domain),
`DD_PRODUCT_CEILING = 60` (1F1 product `w*sqrt(s)`), and
`W_CEILING_SCHWINGER_QD = 150` (hard mpmath upper bound) — all three are
separate variables, none implies the others.

Tests in `test_lensing_schwinger.py`, `test_lensing_batched_operator.py`,
`test_lensing_operator.py`, and `test_lensing_airy_fold.py` updated to
probe `w = 151` (above the QD ceiling) for refusal tests that previously
used `w = 61/80` (now served by the mpmath path).

Commit: `2e387c9`
