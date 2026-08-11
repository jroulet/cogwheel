# Librarian Short-Term Observations

## 2026-08-11 -- bounded fixed-panel GL rule post-commit sync (no-op)

Scope: Post-commit doc sync for commit 823164a "bounded fixed-panel Gauss-Legendre rule for the Schwinger QD band".

### What I found / did

All doc surfaces are clean — this is a no-op run.

- `sync_derived_docs.py` (cogwheel-newlal python): no actual file changes. Only the recurring `lens_amplification_surrogate` test-only-consumer warning (escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md` is open — no duplicate created).
- `overview.rst`: no references to schwinger/mpmath/Gauss-Legendre mechanisms — nothing to update.
- `_schwinger.py` changes: added `_MP_PANEL_ORDER = 32` (private constant), added `_mp_gl_rule()` (private function). No new public API, no serialization artifacts — no changes to api.rst, DATA_CONTRACTS.yaml.
- DOCSTRING CONCERN FROM LAST RUN RESOLVED: `_f_schwinger_mpmath` docstring now correctly names `_MP_PANEL_ORDER` (line 791) — the fix landed in this commit itself. No doc debt remains on that point.
- `git diff --name-only` after sync: only pre-existing agent_state JSON files and memory files — no actual doc changes.

### Patterns / gotchas

- A concern flagged in `librarian_short_term` as "CODE docstring inaccuracy" was resolved IN THE SAME COMMIT that introduced the mechanism — the brief said the build was CODE-complete before I wrote that note, but the doc fix landed in the same feature commit. Pattern: a docstring fix that's part of the same feature commit shows up in the diff but not as a librarian action item.
- No `changelog.d/` directory exists in `.claude/spec/` — changelog entries for internal lensing builds go to `changelog.d/` at repo root (not `.claude/spec/changelog.d/`). The commit already created `changelog.d/2026-08-11_mpmath_fixed_panel_rule.md` directly. This split is confusing but correct per repo convention.

### Cross-references to watch (carried forward)

- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER STILL STALE (INS-1-002/003): SPEC.md ~line 63 and DATA_CONTRACTS.yaml ~line 199 still describe `exterior_polar_rho_log_carrier_v1` as the only known tag. Pending.
- Lobe axis-schema contract (INS-4-002/F050): DATA_CONTRACTS.yaml still describes old lobe axis schemas; production ships `lobe_caustic_relative_v1`. Pending.
- Surrogate escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md`: open, do NOT duplicate.
- SPEC.md and completed.d now cite `_MP_PANEL_ORDER = 32`, `_PANEL_ORDER = 24` (DD path) — both constant-name clusters are fragile.
