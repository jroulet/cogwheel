# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit audit for commits d17f42e and 2e291bc

### Scope

sync_issues.json (which contained d17f42e + 2e291bc) was already DELETED by the
prior librarian run (commit 11b4804: "docs: post-commit sync (render DATA_CONTRACTS_CHANGELOG
after fragment addition)"). The current invocation found no sync_issues.json on disk.

### What was stale

**Nothing.** Both pending commits were resolved:
- d17f42e: handled by prior librarian run (11b4804). DATA_CONTRACTS_CHANGELOG.md was
  rendered and committed.
- 2e291bc (fix(sdk): xdist tree-gate resilience): `.claude/sdk/orchestrator.py` +
  `.claude/sdk/run_full_suite.sh` only — SDK infra, no cogwheel/ code. Zero doc surfaces
  to update per triage rules.

### Files changed

None. No commit made (nothing to fix).

### Uncommitted librarian.json

`.claude/agent_state/librarian.json` is dirty in the working tree: the prior librarian
run updated `last_commit` from 283d435 to 11b4804 but did not include this file in
commit 11b4804. Left uncommitted here too (no doc changes = no commit). Next librarian
run that DOES fix something should include this file in its commit to keep state current.

### Pattern identified

When a sync_issues.json has N commits and the prior librarian run processes all doc-
relevant ones, it deletes the ENTIRE file. If an SDK-only commit appears in the same
queue, it rides along silently (correctly — no action needed). The next librarian
invocation finds no sync_issues.json and confirms via triage that no action was needed.

### Fragile cross-references (continued from prior run)

- `_DD_PRODUCT_MARGIN = 58.0` duplicated in surrogate.py and surrogate_training.py —
  a value change needs both files AND the w-ceiling description in SPEC.md and DATA_CONTRACTS.
- `_FARFIELD_ARC_MAP_SIZE` now used as arc-length map size for wedge tiles too —
  if renamed, the wedge arc-map resolution changes silently.
- `lensing_remaining_coverage_gaps.md` items 1 and 2 remain open `[→ spec]` —
  watch commits touching ppGO paths or `from_wedge_engine` w-ceiling serving logic.
