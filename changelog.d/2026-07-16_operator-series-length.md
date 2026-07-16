---
date: 2026-07-16
---
### Fixed: `F_op` accuracy inside the certified domain (series length + index clamp)

`operator._series_length` sized the kernel series from `|zz| = w*s/2` instead
of the cancellation exponent `L = w*sqrt(s)`; for `|y'| < 1` (the common case)
this under-resolved the derivative ladder and `F_op` returned results accurate
only to ~1e-4 well inside the certified domain. Now sized from `w*sqrt(s)`
(worst-case mpmath-oracle error drops to <= 5.7e-12). Additionally, the dense
contraction's fancy-index ran off the derivative ladder for zero-coefficient
table corners, raising `IndexError` on every `F_op` call with `max_order >= 1`;
out-of-range cells are now clamped (they carry zero coefficients, so the
clamped lookup is multiplied away). Both found by the independent test suites'
mpmath oracles. See FINDINGS F005 for the remaining open gap at `L > ~30`.
