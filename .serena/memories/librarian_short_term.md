# Librarian Short-Term Observations

## 2026-08-07 m_lens_range post-commit sync (no-op)

**Scope**: post-commit triggered by commit 6144378 (`feat(lensing): m_lens_range
override for train()/PriorBox`). Changed files: `cogwheel/lensing/surrogate_training.py`,
`CHANGELOG.md`, `changelog.d/2026-08-07_train_m_lens_range.md`, memory files,
`agent_state/librarian.json`.

**Result**: clean no-op. All doc surfaces were already current from the PRIOR
librarian run (same session, code was uncommitted then; now committed). Nothing
new to do:

- SPEC.md TRAINING paragraph: does not cite `train()` keyword options — confirmed
  (grep of `m_lens_range|from_prior_classes` hits only TODO.md and a TODO fragment,
  not SPEC.md prose).
- DATA_CONTRACTS.yaml: `lens_amplification_surrogate` producer is the script entry
  point, not the training signature; no update needed.
- `overview.rst`: lensing section covers engine + waveform + likelihood only; no
  reference to `train()` or `PriorBox` signatures. Confirmed by pattern search.
- `crash_course.rst`, `api.rst`: no relevant changes.
- CHANGELOG.md: already updated (fragment written and rendered in prior run).

**sync_derived_docs.py output**: same recurring four test-only-caller warnings for
`lens_amplification_surrogate`. Escalation TODO (`todo.d/surrogate_contract_test_
consumer_warning.md`) already exists — confirmed present, no new fragment needed.
"Some issues auto-fixed" message was an internal state flush (no new git diff).

**Stray diff**: `.claude/tidy_advisory.json` was already dirty before this run
(per initial git status `M .claude/tidy_advisory.json`) and was not committed —
left as-is per the known tidy_advisory side-effect pattern.

**Pattern**: m_lens_range type of training-option changes will reliably be no-ops
on all doc surfaces. SPEC.md's TRAINING section documents pipeline mechanics
(band structure, tiling, registration gate), not `train()` keyword signatures.
New optional keyword arguments with `None` defaults are invisible at the SPEC level.
