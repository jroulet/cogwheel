---
date: 2026-08-04
section: Lensing / Surrogate
---

### Cusp arm coverage enabled (`_CUSP_ARM_COVERAGE = 0.07 rad`)

`_CUSP_ARM_COVERAGE` pinned to 0.07 rad by direct arm boundary measurement
(`scripts/measure_cusp_arm_actual_boundary.py`): minimum angular offset from
the cusp vertex at which `cusp_amplification` serves, across gamma=[0.1..1.5],
w=[10..40], floored to 2 decimal places (conservative).

The Pearcey table (shipped in c715bcd) + this coverage constant means cusp-window
draws within 0.07 rad of the cusp vertex are now served by the Pearcey arm instead
of falling through to exact quadrature. The tube's exclusion window shrinks by
`_CUSP_ARM_COVERAGE` at query time (not stored in chart schema). Residual draws
beyond the arm's certified reach still fall through to the exact engine.

Inspector: PASS. Professor: PASS. Certified by `test_lensing_cusp_arm_coverage.py`.
