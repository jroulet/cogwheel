# Inspector Short-Term Observations

## Review: Build 3 (C6) — Test Compatibility Port + Production Curvature-Relative Tube (FINAL PASS)
**Date**: 2025-07-XX (current)
**Scope**: Test files (4 planned + 1 extra: caustic_cusps) ported to new `f_max`/`f_floor`
TrainingConfig fields; production surrogate_training.py replaces `eta_max`/`eta_floor` fields
with curvature-relative `f_max`/`f_floor`; surrogate.py gains lobe sqrt-edge s-coordinate;
test_lensing_surrogate_lobe.py and scripts/measure_tube_fraction.py also fixed.

## Findings

### ALL THREE PREVIOUS FINDINGS RESOLVED

- INS-1-001: test_lensing_surrogate_lobe.py now passes `eta_max=_LOBE_ETA_MAX` at both call
  sites (lines 139-140 and 778-779). Confirmed green: 54 passed.
- INS-1-002: scripts/measure_tube_fraction.py removed `dataclasses.replace` call, now passes
  `eta_max`/`eta_floor` as explicit kwargs to `_build_tube_chart` and `_tube_heldout_samples`.
  Syntax verified.
- INS-1-003: Comment at CLEARANCE_SLACK now says "4001-point" (matching INTERIOR_DENSE_SAMPLES).

## Test Results (all green)
- test_lensing_exterior_admission.py: 42 passed
- test_lensing_exterior_windows.py: 76 passed, 1 xfailed
- test_lensing_ppgo_bandsplit.py: 62 passed, 4 skipped
- test_lensing_surrogate_training.py: 31 passed, 49 skipped
- test_lensing_surrogate_lobe.py: 54 passed
- test_lensing_caustic_cusps.py: 42 passed

## SPEC staleness (carried, flag to Librarian)

SPEC.md TRAINING paragraph still says "a runtime FOOT-OF-NORMAL guard skips a tube chart
whose `eta_max` exceeds half the band's minimum caustic curvature radius". The code now uses
a curvature-relative design with `assert config.f_max < 0.5` — no skipping ever occurs.
This is a Librarian task to update.

## Passed checks
- Production import correctness: both surrogate.py and surrogate_training.py import cleanly.
- Caller/callee consistency: all callers of `_interior_admission`, `_saddle_lobe_admissions`,
  `_build_tube_chart`, `_tube_heldout_samples` verified via `find_referencing_symbols` —
  all pass the new `eta_max` (and `eta_floor` where needed) as explicit keyword arguments.
- No remaining `config.eta_max` or `config.eta_floor` references in any test or script.
- Lobe sqrt-edge coordinate: training, serve, and serialization logic are internally consistent.
- V1 backward-compat: V1 schema (theta_to_s=None) handled in load and evaluate.
- DATA_CONTRACTS: already updated for both lobe axis schema tags.
- No new code without tests (new test classes in test_lensing_caustic_cusps.py cover
  the curvature-relative invariant).

## Open issues (Librarian only)
- SPEC.md training paragraph needs foot-of-normal text replaced with curvature-relative
  description.
