---
date: 2026-08-09
section: Backlog
---

### High-w cusp serving via ppGO instead of Pearcey live quadrature

`cusp_amplification` now opens with a high-w ppGO fast rung
(`cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`): when the control radius
clears the ppGO gate and `w >= _W_PPGO_FLOOR`, it serves
`_airy_fold.fold_ppgo_correction` directly — the Pearcey function's
geometric-image-sum limit — a ~10^3x faster path that returns before any
table or quadrature lookup. On rung refusal (geometry `LensDomainError` or a
non-finite result) it falls through to the uniform Pearcey path unchanged,
preserving the None-fall-through contract. The rung applies to both the
astroid (positive-parity) and saddle cusp branches.

Three module-level constants ship with the rung: `_R_PPGO_ERROR_CONST = 50.0`
(the leading-error coefficient for the ppGO gate), `_W_PPGO_FLOOR = 50.0`
(kernel-truncation floor), and `_PPGO_BAR_DIVISOR = 10` (the ppGO envelope
bar divisor, tightening the bar relative to the Pearcey uniform-form gate).

Tests: 5 classes / 13 tests added in `test_lensing_airy_fold.py`
(`PpgoGoldenAgreementTestCase`, `PpgoRungRefusalTestCase`,
`PpgoFinitenessGuardTestCase`, `PpgoSaddleParityTestCase`,
`PpgoRungSelfFalsificationTestCase`). Inspector PASS (INS-6-001, INS-6-002
resolved, no findings).

Driver post-build verification owed: `_R_PPGO_ERROR_CONST = 50.0` is
provisional — the constant is documented as owed a post-build driver
measurement to tighten the cross-over radius against the exact Pearcey
reference.
