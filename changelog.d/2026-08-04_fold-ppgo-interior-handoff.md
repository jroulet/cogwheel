## 2026-08-04

### Lensing: fold-ppGO interior handoff serve path

Interior positive-parity draws (`rho <= 1.0`) that fall above the
`InteriorWedgeChart` DD-product w-ceiling — where no trained chart covers
the point — are now served directly by
`likelihood._surrogate_coefficients` (the "fold-ppGO handoff") when two
conditions are simultaneously met:

1. The merging fold pair is well-resolved:
   `xi_min = (3 * w_min * Delta_tau / 4)^{2/3} >= _XI_FOLD_THRESHOLD = 4.0`
2. The per-pair uniform error estimate (`_uniform_error_estimate`) is below
   `CERTIFICATION_BAR`.

Reconstruction mirrors the Born rung path (`reconstruct_farfield` with
`FARFIELD_KERNEL_SUM`).  Draws failing either gate still fall through to
the exact engine as before.

`surrogate_census.characterize_sample` mirrors the same gate and records
the new category `'ppgo_fold'` for qualifying draws, extending the census
breakdown from 6-way to **7-way** MECE
(`gamma-guard / dropped-sliver / born / cusp-window / refusal-ball /
out-of-box / ppgo_fold`).

**New test file:** `cogwheel/tests/test_lensing_fold_ppgo_handoff.py`
(897 lines; 10 tests covering accuracy vs exact engine at high xi, xi gate
refusal for near-caustic sources, envelope reconstruction round-trip,
self-falsification witnesses, c_A fine gate refusal, default path
unaffected, census `ppgo_fold` recording).

Commit: `d9e88a2`
