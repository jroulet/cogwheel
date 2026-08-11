# Librarian Short-Term Observations

## 2026-08-11 -- post-commit sync for 3de70a2 (NO-OP)

Scope: docs: todo + brief for revert-residual-table / fix-cusp-routing build.

Changed files in trigger commit:
- `.claude/handoff/brief_revert_residual_table_fix_routing.md` — agent-only path
- `.claude/spec/todo.d/lensing_revert_residual_table_fix_routing.md` — todo fragment

No `cogwheel/` Python files changed. No new modules, no new disk artifacts, no public API
changes. `TODO.md` was already regenerated as part of the commit.

Also checked `bed77b8` (style: remove shadowed `_KNOWN_ENVELOPE_DEFINITIONS` alias from
`cogwheel/lensing/surrogate.py`): the alias name appears only in `completed.d/` history,
not in any living doc surface. NO-OP.

### FOLD-CARRIER SCHEMA CROSS-REF (INS-1-002/003) — NOW CONFIRMED FIXED

Previous short-term memory carried this as "Still pending" — WRONG. Both surfaces are
already correct:
- SPEC.md ~line 62: "tag `'exterior_polar_rho_log_carrier_v1'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V4`)
  — retained for backward compatibility — and the current write tag
  `'exterior_polar_rho_u_carrier_v2'` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V5`) are the two
  known tags"
- DATA_CONTRACTS.yaml line 198: "Each such record MUST carry one of the two known
  axis_schema tags: 'exterior_polar_rho_log_carrier_v1' ... or
  'exterior_polar_rho_u_carrier_v2' ... the current write tag"
Neither says "ONLY known tag" for V4 anymore. Fix was applied in a prior session not
captured in the short-term memory. DO NOT re-apply this fix.

### Lobe axis schema INS-4-002 — CONFIRMED CORRECT

DATA_CONTRACTS.yaml already correctly describes `lobe_caustic_relative_v1` as the ONLY
known lobe tag, with the two old lobe tags dropped. Nothing stale here.

### sync_derived_docs.py

Ran clean (5 checks). Test-consumer warnings for `lens_amplification_surrogate` recurred
identically — the escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md`
still exists and is open. DO NOT create a duplicate. "Auto-fixed" claim was the known
false-positive state flush; `git diff` showed only pre-existing dirty agent state, no doc
surface changes.

POST-COMMIT SYNC NO-OP RULE applies: no doc surfaces were stale, no files committed
(beyond sync_issues.json deletion + this memory write).
