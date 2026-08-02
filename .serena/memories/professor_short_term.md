# Professor Short-Term Observations

## Build review: TrainingConfig eta_max/eta_floor field removal PORT (2026-08-XX)

Four test files ported to use explicit `eta_max`/`eta_floor` arguments instead of
the retired `TrainingConfig.eta_max`/`TrainingConfig.eta_floor` fields:

| File | Tests | Result | Runtime |
|------|-------|--------|---------|
| test_lensing_exterior_admission.py | 42 | 42 passed | 3m09s |
| test_lensing_exterior_windows.py | 77 | 76 passed, 1 xfailed | 2m54s |
| test_lensing_ppgo_bandsplit.py | 66 | 62 passed, 4 skipped (TRAIN_TIER) | 27s |
| test_lensing_surrogate_training.py | 80 | 31 passed, 49 skipped (TRAIN_TIER) | 6s |

Zero import errors, zero AttributeErrors, zero failures.

### Key verifications:
- Renamed test methods (`test_f_max_matches_training_config`,
  `test_f_max_constant_matches_training_default`) confirmed running and passing.
- All saddle-lobe admission tests pass (centroid/corridor/winding/Morse/cusp-aligned).
- No residual references to `config.eta_max` or `config.eta_floor` remain in any of
  the four files (grep-verified zero hits).
- No `dataclasses.replace(..., eta_max=...)` patterns remain in the ported files.
- Module-level constants `_WP3_ETA_MAX=0.05`, `_WP3_ETA_FLOOR=0.02`,
  `_PPGO_ETA_MAX=0.05`, `ETA_MAX=0.05`, `SADDLE_ETA_MAX` correctly threaded to all
  call sites.
- 49 skipped tests in surrogate_training are all `COGWHEEL_TRAIN_TIER=1` gated
  (engine-backed, minutes/class) — expected skip for fast-tier.
- Diagnostic plots produced (20 PNGs in `cogwheel/tests/output/`).

### Physics correctness notes:
- The `eta_max` parameter controls the tube-shell exclusion radius in the
  Chang-Refsdal lensing surrogate exterior/interior admission logic. Moving it
  from the `TrainingConfig` dataclass to an explicit function argument is a pure
  refactoring of the call interface — no change to the underlying geometry or
  admission predicate.
- The fixed value 0.05 (in Einstein-radius units) remains the standard operating
  point for tube-shell geometry at the default f_max=0.40 design point.
- The saddle lobe admission (`_saddle_lobe_admissions`) passes through correctly
  with explicit `eta_max=SADDLE_ETA_MAX`.

### Deferred:
- TRAIN_TIER=1 tests (heavy engine-backed surrogate builds) are operator-deferred
  ship gate, not runnable in a fast-test budget.

**Verdict: PASS** — all fast tests green, interface port correct, no physics changes.
