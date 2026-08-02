# Inspector Short-Term Observations

## Review: 2026-08-01 — Envelope-extrapolation fallback for interior ppGO cells (pass 3)

### Scope
- Production: `cogwheel/lensing/ppgo_map.py` — new `_extrapolate_floor()` function + integration into `_measure_cell()`.
- Tests: new `cogwheel/tests/test_lensing_extrapolate_floor.py` (10 fast-tier tests pass, 4 TRAIN_TIER-gated skip).
- Existing suite: 62 tests in `test_lensing_ppgo_bandsplit.py` all pass unchanged.

### Previously Open Findings — Re-checked

1. **INS-1-001** (unreachable `C <= 0.0` guard): STILL PRESENT. `C = math.exp(coeffs[1])` is always positive. The guard is defensive dead code. Trivial.

2. **INS-1-002** (DATA_CONTRACTS empty-range semantics): STILL PRESENT. Neither SPEC.md nor DATA_CONTRACTS.yaml were modified. The new empty-range certified-interior-cell case is not documented in contracts. Flag to Librarian.

3. **INS-1-003** (misleading `_EXTRAP_W_CERT_DEFLATION` name): STILL PRESENT. The constant still says "deflation" when the operation inflates the floor. Trivial.

### New Findings

None. The code is correct:
- Math verified: power-law fit in log-log, extrapolation formula `(C/bar)^(1/alpha)` is correct.
- R² computation is correct: `np.var(log_err) * len(log_err)` = sum of squared deviations = ss_tot.
- Guards (alpha bounds, R² threshold, max extrapolation ratio, deflation safety factor) are properly gating.
- The `floor > w_ceiling` relaxation for interior cells is SAFE: downstream consumer `_surrogate_coefficients` either refuses the band-split via `w_hi > eff_ceiling` or because `w_trust > w_hi` (since w_trust > w_cert > w_ceiling), so no incorrect serving occurs.
- `_compute_interpolable` handles the large-jump case conservatively (marks cell non-interpolable).
- Import correctness verified.
- All fast-tier tests pass with anti-vacuity and self-falsification.
- TRAIN_TIER tests are correctly gated and use `skipTest` for configs whose envelope shape may not be clean enough.
- Existing TruncationOnRefusalTestCase (RHO_CENTER=0.5, interior) still passes correctly because the stub produces zero error everywhere on the accepted prefix, so `_sup_over_w_floor` returns the first node (not None), and the extrapolation fallback never triggers.
- per_angle_data indexing is safe: the loop has early returns for best_k==0 or zero denominator (returning STATUS_INVALID), so the extrapolation fallback only runs when all 9 angles have corresponding data.

### Correctness Assessment
- The `_EXTRAP_MAX_RATIO` check before deflation is intentionally correct: it gates the PHYSICS extrapolation trustworthiness, then the deflation adds a safety margin on top.
- The `per_angle_data` storage uses the truncated `w_prefix` (accepted prefix), which is correct for fitting the tail of whatever data was measured.
- No behavioral change for exterior cells (rho_center >= 1.0): the `rho_center < 1.0` guard ensures only interior cells enter the extrapolation path, and the `floor > w_ceiling and rho_center >= 1.0` guard preserves the old refusal for exterior cells.

### Open Issues Carried Forward
- INS-1-001, INS-1-002, INS-1-003 (all trivial/Librarian scope)
