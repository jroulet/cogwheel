# Inspector Short-Term Observations

## 2025-07-28: Build fix_dropped_slivers review (re-review pass 2)

### Scope
WP1: reduce `min_gamma_band` from 0.02 to 0.005 in `TrainingConfig` and
`stable_gamma_bands` default. Also updates `scripts/measure_dropped_slivers.py`
constant and docstrings in `surrogate_training.py`.

### Findings
- **PASS** — Production code change is a single-constant default-value change
  at two sites (dataclass field + function signature default), consistent with
  each other and with the `train()` call site which passes `config.min_gamma_band`.
- Existing tests (`StableGammaBandsF041TestCase`) pass because they explicitly
  specify `min_width=0.02` in every call, so they're unaffected by the default change.
- The new test suite (`test_lensing_min_gamma_band.py`) has 9 tests, all pass
  (21.0s total). It tests: fewer dropped at 0.005 vs 0.02, all remaining drops
  are < 0.005, mutation check, mocked threshold boundary, and self-falsification.
- `scripts/verify_coverage.py` uses the default and will now be more permissive
  (fewer dropped slivers = more stable bands found). This is correct behavior.
- **NOTE (non-blocking, INS-1-001 carried forward)**: The brief's acceptance
  criterion ("REGION 10 CLOSED", dropped fraction < 1e-3) is NOT achieved —
  measured result is 5.36e-3. This is a brief estimation error (the remaining
  slivers are genuine topology-straddling bands narrower than 0.005), not an
  implementation defect. The implementation correctly achieves what was asked
  (reduce threshold to 0.005). The prior mass dropped was reduced from 2.15%
  to 0.54%.
- Docstring edits ("far-field/exact serving" → "no chart and fall through to the
  exact engine") are more accurate descriptions of the behavior.
- SPEC.md references `min_gamma_band` as a concept without pinning a specific
  numeric value — no spec divergence from the code change.
- No DATA_CONTRACTS.yaml impact.
- All callers of `stable_gamma_bands` traced via semantic tools: production (1
  call site in `train()`), existing tests (F041 suite, 6 call sites, all pass
  explicit min_width=0.02), new test file (9 tests, all pass), scripts (2: 
  measure_dropped_slivers with explicit MIN_WIDTH=0.005, verify_coverage using
  default — both correct).

### No new patterns discovered.
### Open issues carried forward:
- INS-1-001: brief's acceptance criterion gap (not a code defect, brief estimation error).
