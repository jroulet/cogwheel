# Inspector Short-Term Observations

Date: 2026-08-11
Scope: Build "operator_routing_one_home" — WP1 ppGO resolution gate in `_pearcey_cusp.py`

## Diff (production)
- `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`: added `_PPGO_RESOLUTION_GATE = 4.0` (mirrors `operator.RHO_END`, circular import prevents direct import) and dual gate `_merging_fold_pair is not None OR w * delta_min >= _PPGO_RESOLUTION_GATE` in the ppGO rung block of `cusp_amplification`.
- `cogwheel/tests/test_lensing_airy_fold.py`: added `test_resolution_gate_isolated_admit_and_refuse` (4 checks) to `PpgoRungSelfFalsificationTestCase`.

## Implementation review

### Correctness
- Dual gate correctly separates fold-pair nodes (Morse 0,1: `fold_ppgo_correction` valid regardless of resolution) from saddle-only nodes (Morse 2,3: only valid above `w*delta_min >= 4.0`).
- `_PPGO_RESOLUTION_GATE = 4.0` matches `operator.RHO_END = 4.0`.
- `delta_min` computed from pairwise sorted delays; `len(delays) < 2` → 0.0 (conservative).
- `_merging_fold_pair` inside try/except (can raise LensDomainError) — correct.
- Fold-pair and resolved saddle nodes are BYTE-IDENTICAL to pre-change behavior.
- Only unresolved saddle nodes (the brief's failing configs: `w*delta_min ≈ 1.90 < 4.0`) are newly refused — correct physics.

### Test results
- `test_thresholds_have_one_home`: PASSED (this was the named failing test at HEAD)
- Full gate: 128 passed, 11 skipped, 2 xfailed (operator + fast_path + airy_fold)
- All ppGO tests pass: PpgoGoldenAgreementTestCase (2), PpgoRungRefusalTestCase (3), PpgoFinitenessGuardTestCase (4), PpgoSaddleParityTestCase (2), PpgoRungSelfFalsificationTestCase (3 including new)
- Self-falsification test has teeth: raising gate to 1000 blocks, lowering to 0 admits, resolved w still admits.

### Findings: NONE
All acceptance criteria met. The implementation is refusal-conservative, physics-justified, and tests are value-asserting with reachable-red.

## Pre-existing findings carried forward (not from this diff)
INS-1-002/003 (exterior_polar_rho_log_carrier_v1 doc staleness), INS-1-001 (SPEC.md saddle exterior stale).

## Pre-existing test failures (not from this diff)
8 vertex-related tests in test_lensing_airy_fold.py fail at HEAD due to _cusp_vertex behavior change from a different build (WP1 _cusp_vertex routing fix in Build 2026-08-11, already committed). Not in scope for this review.
