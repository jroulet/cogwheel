# Inspector Short-Term Observations

## Review: 2026-08-02 — Fold-corrected ppGO (brief_fold_corrected_ppgo, WP1+WP2) — PASS REVIEW

### Scope
- Production: `cogwheel/lensing/chang_refsdal/_airy_fold.py` — new `fold_ppgo_correction` function (public, exported); relaxation of `_uniform_error_estimate` at xi=0.
- Production: `cogwheel/lensing/ppgo_map.py` — `_measure_cell` now uses `fold_ppgo_correction` instead of `geometric_amplification`.
- Production: `cogwheel/lensing/chang_refsdal/channels.py` — `born_carrier_from_partition` gains inline fold correction block (positive-parity above-split only).
- Production: `cogwheel/lensing/chang_refsdal/__init__.py` — exports `fold_ppgo_correction`.
- Tests: new `test_lensing_fold_ppgo_correction.py` (23 tests pass, ~5.2 s).
- Tests: `test_lensing_ppgo_bandsplit.py` — patch added for `fold_ppgo_correction` (62 passed, 4 skipped).
- SPEC.md: trivial label rename ("far-annulus carrier" → "exterior rung carrier").

### Findings: NONE (all clean)

### Previously Open Findings — RESOLVED
1. **INS-c8-001**: RESOLVED. The test now patches `_airy_fold_module.fold_ppgo_correction` alongside `geometric_amplification`, ensuring the zero-error invariant holds. Verified: 62 passed, 4 skipped.
2. **INS-c8-002**: RESOLVED. The inner variable is now `fold_delta_tau` (line 1604), not `delta_tau`.
3. **INS-c8-003**: RESOLVED (accepted by design). Both locations carry cross-reference `# See INS-c8-003` comments documenting the intentional duplication and the contract that both must be updated together.

### Correctness Assessment
- The math is correct: `xi = (3*w*Delta_tau/4)^{2/3}` matches the spec; the correction `full_ppgo - pair_ppgo + airy_values` correctly replaces the pair's contribution while keeping all other images intact.
- Frame arithmetic is correct: `born_carrier_from_partition` demodulates from absolute to min-relative frame via `exp(-1j*w*t_min)`.
- Structural gates are conservative: any gate refusal falls back to raw ppGO (byte-identical or no-op).
- The DO-NOTHING property (no ETA gate) matches the brief's design intent.
- The xi=0 relaxation in `_uniform_error_estimate` is physically correct (Airy exact on fold).
- The `__init__.py` export is clean and matches `__all__`.
- The new test suite is well-structured with anti-vacuity, self-falsification, and non-circular oracles (Schwinger).
- Exception coverage is complete: `LensDomainError`, `ValueError`, `ZeroDivisionError` all caught in the channels block.
- Mock patching in bandsplit test works correctly (patches module attribute before the function-local `from ... import` executes).
- All related test suites pass: 126 passed, 8 skipped, 1 xfailed.

### Open Issues Carried Forward
- INS-1-001 (unreachable `C <= 0.0` guard in ppgo_map.py): STILL PRESENT. Trivial.
- INS-1-002 (DATA_CONTRACTS empty-range semantics): STILL PRESENT. Trivial / Librarian scope.
- INS-1-003 (misleading `_EXTRAP_W_CERT_DEFLATION` name): STILL PRESENT. Trivial.
