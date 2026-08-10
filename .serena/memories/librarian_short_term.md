# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: tidier pass on 2D fold-carrier (commit 992c500)

### Scope
Post-commit mode triggered by `.claude/sync_issues.json` for commit 992c500
(style: tidier pass on 2D fold-carrier build — remove unused dimensionless_frequency
import from `cogwheel/tests/test_lensing_surrogate_training.py`).

### Outcome: NO-OP — test-only change, skip entirely
Changed file is a test file only. Triage rule: "Notebook or test-only changes → skip entirely."
No doc surfaces stale. No sync script needed.

### Fragile cross-references (carried forward from previous sessions)
- Both SPEC.md and DATA_CONTRACTS.yaml cite `_EXTERIOR_POLAR_AXIS_SCHEMA_V4`,
  `_EXTERIOR_POLAR_AXIS_SCHEMA_V5`, and the two literal tags
  (`exterior_polar_rho_log_carrier_v1`, `exterior_polar_rho_u_carrier_v2`)
- SPEC.md cites `_compute_rho_u_carrier` — rename touches both surfaces
- "Old 1-D rho_carrier artifacts load by broadcasting to 2-D" sentence paired with
  V4-retained claim — if V4 is ever dropped or broadcast removed, all three sentences
  go stale together
- `todo.d/surrogate_contract_test_consumer_warning.md` remains open (escalation
  fragment exists; do NOT create a duplicate)

### Pre-existing stray diff
`tidy_advisory.json` and `tidy_short_term.md` had pre-existing modifications at session
start — NOT caused by this session, not staged.
