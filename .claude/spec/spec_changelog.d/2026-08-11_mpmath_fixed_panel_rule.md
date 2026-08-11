---
date: 2026-08-11
bump: patch
---
### Schwinger QD band: fixed-order composite Gauss-Legendre rule at mpmath precision

Updated the `_schwinger.py` description in the Microlensing engine row (Key
abstractions): for `60 < w <= W_CEILING_SCHWINGER_QD = 150`, the mpmath path
(`_f_schwinger_mpmath`, `dps = 30 + ceil(w)`) now uses a fixed-order
composite Gauss-Legendre rule at `_MP_PANEL_ORDER = 32` per panel (replacing
the adaptive per-panel `mp.quad`), with the N/2N paired certification
computed on the mpmath-reconstructed F. The band is bounded and
deterministic, and order-32 preserves the `_CERTIFICATION_TOL = 3e-10`
certification bar across the band (serving coverage in the cusp-exterior
windows is unchanged).
