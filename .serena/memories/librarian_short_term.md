# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit sync for commit 9cefb88 (finding: d-axis normalization evaluated and rejected)

### Scope

sync_issues.json listed one pending commit: 9cefb88 "finding: d-axis normalization evaluated and rejected".

Changed files in that commit:
- `.claude/spec/completed.d/2026-08-03_d_normalization_evaluation.md` — added
- `.claude/spec/todo.d/lensing_remaining_coverage_gaps.md` — updated (d-norm item marked RESOLVED)
- Agent state and memory files (no doc action needed)

### What was stale

1. **COMPLETED.md** — not regenerated after `completed.d/` fragment was added. Fixed by
   running `render_fragments.py` (now includes d-normalization evaluation entry).
2. **TODO.md** — not regenerated after `todo.d/lensing_remaining_coverage_gaps.md` was updated
   to mark d-normalization as RESOLVED. Fixed by same `render_fragments.py` run.
3. **FINDINGS.md** — the commit was titled "finding:" and the evaluation produced a
   permanent architecture insight (d/R_c normalization is wrong physics, wrong chart,
   breaks separability). Added F060 directly to FINDINGS.md.

### Files changed in this commit

- `.claude/spec/COMPLETED.md` — added d_normalization_evaluation entry from completed.d fragment
- `.claude/spec/FINDINGS.md` — added F060 (d-axis normalization evaluation/rejection)
- `.claude/spec/TODO.md` — d-norm item now shows as RESOLVED/strikethrough
- `.serena/memories/librarian_short_term.md` — this file
- `.claude/sync_issues.json` — deleted (trigger file consumed)

### Pattern

"finding:" commits that place their fragment in `completed.d/` (not `findings.d/`) create
THREE stale surfaces simultaneously: COMPLETED.md (not re-rendered), TODO.md (item not yet
resolved in generated file), and FINDINGS.md (physics/design insight not recorded as a finding).
All three need attention on the same sync pass.

The prompt's diff stat showed `findings.d/` for this fragment; `git show --name-only` confirmed
it was actually `completed.d/`. Trust `git show --name-only` over the diff stat in the brief.

### sync_derived_docs.py output

Reported 4 test-file-only callers of `LensAmplificationSurrogate.load` not in
DATA_CONTRACTS.yaml consumers list. Per convention (and librarian_knowledge), test-file-only
callers are excluded — no action taken.

### Fragile cross-references (continued)

- F060 cites `cogwheel/lensing/surrogate.py` for the far-field chart d-axis — watch for
  module reorganization.
- `_DD_PRODUCT_MARGIN = 58.0` still duplicated in surrogate.py and surrogate_training.py.
- `lensing_remaining_coverage_gaps.md` items 2 and 3 remain open (`[→ spec]` and
  `[research]`) — watch commits touching ppGO paths or interior cell certification.
- The `[housekeeping]` items in lensing_remaining_coverage_gaps.md (sidecar callback
  and xdist tree-gate) have no `[→ spec]` tag — they don't drive doc updates when closed.
