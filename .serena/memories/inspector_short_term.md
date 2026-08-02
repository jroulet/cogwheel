# Inspector Short-Term Observations

## Review: Fix CarrierDiscontinuityError in _reprovision_w_nodes (2026-08-01)

### Scope
- `cogwheel/lensing/surrogate_training.py`: +5 lines (except clause in `_eps_for`)
- `cogwheel/tests/test_lensing_exterior_windows.py`: +54 lines (new test method)

### Findings
- **PASS**: No issues found.
- Production fix adds a `CarrierDiscontinuityError` catch in `_eps_for` (nested
  in `_reprovision_w_nodes`), consistent with the existing pattern at all other
  `_build_farfield_chart` call sites (lines 3748, 4436 of same file).
- Test is a well-structured fault-injection test (mock raises the exception,
  verify it doesn't propagate, verify correct status/trace entries).
- Exception ordering correct: `_ENGINE_REFUSALS` does NOT include
  `CarrierDiscontinuityError` (it's ValueError but not LensDomainError/
  SchwingerCertificationError/HypergeometricDomainError), so a separate
  except clause is necessary.
- No data contract changes needed (in-memory error handling only).
- Spec-consistent: SPEC.md explicitly documents `_build_farfield_chart` raising
  `CarrierDiscontinuityError` and callers handling gracefully.
- All 4 tests in `ReprovisionNodeCountTestCase` pass green.

### Open issues carried forward
- None from this review.
