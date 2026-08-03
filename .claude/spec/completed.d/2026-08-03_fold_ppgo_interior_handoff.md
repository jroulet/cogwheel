---
date: 2026-08-03
section: Likelihood
---

### Fold-ppGO interior handoff above InteriorWedgeChart w-ceiling

Added a new serve path in `likelihood._surrogate_coefficients` for positive-parity
interior draws (`rho <= 1.0`) above the `InteriorWedgeChart` DD-product w-ceiling.
When the merging fold pair satisfies `xi_min = (3*w_min*Delta_tau/4)^(2/3) >= 4.0`
(geometric resolution gate) AND the per-pair uniform error estimate
(`_uniform_error_estimate`) is at or below `CERTIFICATION_BAR`, reconstruction
proceeds via `fold_ppgo_correction` with `reconstruct_farfield`/`FARFIELD_KERNEL_SUM`,
mirroring the Born rung path. Draws failing either gate fall through to the exact
engine as before.

`surrogate_census.characterize_sample` mirrors the same gate logic and records
category `ppgo_fold` for qualifying interior draws, extending the census breakdown
from 6-way to 7-way.

Certified by `cogwheel/tests/test_lensing_fold_ppgo_handoff.py` (17 tests across
3 phases: accuracy vs exact engine, xi gate refusal, envelope reconstruction
round-trip, self-falsification, c_A fine gate refusal, default-path unaffected,
census `ppgo_fold` recording).

SPEC.md updated: FOLD-PPGO INTERIOR HANDOFF paragraph added to the "Microlensed
waveform & likelihood" row; census breakdown updated to 7-way.
