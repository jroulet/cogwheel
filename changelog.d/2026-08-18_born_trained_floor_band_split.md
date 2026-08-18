---
date: 2026-08-18
---

### Born far-field: trained-floor band split revives covered-but-below-floor draws

`_born_residual_analytic` no longer refuses a whole band to the exact
engine just because the host sub-band's low edge falls below the
Born residual chart's trained `log_w` floor. When the box covers the
host, the chart now serves the trained sub-band it was actually built
for, and the exact engine hosts only the untrained remainder below it.
Measured on a 10k-draw engine-free census: 3.43% of draws move from
`engine_residual`/`diffractive_analytic` to `born_analytic`.
