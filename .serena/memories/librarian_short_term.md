# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for 897bff8 + 26d088a + cfc4377

**Scope**: three commits in the sync_issues.json queue

**Outcome**: NO-OP — no doc surfaces were stale.

**Per-commit triage**:

1. **897bff8** ("docs: post-commit sync (c8cad0c deltoid exterior cusp gap + mid-w ppGO band)"): This IS the prior doc-sync commit — already fully propagated in the previous librarian session (see prior short_term). Nothing new to propagate.

2. **26d088a** ("test: consolidate duplicate routing-path pins [housekeeping]"): Test-only change (test_lensing_airy_fold.py). Skipped per triage rules.

3. **cfc4377** ("docs: retire duplicate-routing-pins consolidation"): Housekeeping — moved `tests_consolidate_duplicate_routing_pins.md` from todo.d to completed.d, regenerated COMPLETED.md/TODO.md. No Sphinx docs, SPEC prose, or DATA_CONTRACTS stale.

**sync_derived_docs.py**: 5 checks run. The recurring `lens_amplification_surrogate` test-consumer warning appeared again — still covered by the open `surrogate_contract_test_consumer_warning.md` TODO fragment. No action taken. No real git diff produced.

**render_fragments.py**: "All surfaces up to date." No fragments to render.

**What was NOT stale**:
- overview.rst, api.rst, crash_course.rst, installation.rst
- SPEC.md prose (feature changes already propagated in 897bff8)
- DATA_CONTRACTS.yaml (no new disk artifacts)
- CHANGELOG.md (already updated in 897bff8)

**Fragile cross-references to watch** (inherited from prior session):
- SPEC.md CUSP-EXCLUSION FILTER "for positive parity (astroid)" — stale if astroid behavior also changes
- `FARFIELD_KERNEL_SUM_MINUS_GHOST` label in SPEC.md certified-by — stale if label renamed
