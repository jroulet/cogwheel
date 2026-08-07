# Librarian Short-Term Observations

## 2026-08-07 post-commit sync (commit e4b7b80)

**Scope**: feat(lensing): regions filter for training + slow-operation admission judge.

**Changed cogwheel/ files**: `cogwheel/lensing/surrogate_training.py` (new `regions=` parameter
to `train()` and `_train_band_charts()`; new public `guard_slow_operation()` function;
new private `_self_estimate()` function). `scripts/train_lens_surrogate.py` (+`--regions` CLI flag).
Test files skipped per rules.

**Serialization check**: no new disk artifacts. All new code is parameter-filtering logic
(regions filter) and an in-memory admission judge. DATA_CONTRACTS.yaml unchanged.

**SPEC.md check**: TRAINING paragraph describes architecture, not signatures. The `regions`
parameter and `guard_slow_operation` are implementation-level — nothing to propagate.
`guard_slow_operation` is also not a Sphinx-doc-surfaced API concern (module docstring carries it).

**TODO closure**: `lensing_training_path_cannot_be_run_per_region.md` closed — the
implemented regions filter is exactly what the fragment prescribed. Moved to
`completed.d/2026-08-07_lensing-training-path-per-region.md`. Two todo.d fragments that
had `[[lensing_training_path_cannot_be_run_per_region]]` wiki-links were repointed to the
completed.d record:
- `lensing_chart_kinds_should_share_one_tiling_machine.md` (item b, parenthetical)
- `lensing_wedge_probe_charts_need_retraining_under_v3.md` (final paragraph)

**What went stale and why**: The TODO fragment wasn't touched by the feature commit (normal —
feature commits don't typically close their own todo.d fragments). Required a post-commit
librarian pass to detect the closure and repoint wiki-links.

**Consumer-graph warning**: `surrogate_contract_test_consumer_warning.md` TODO fragment
already exists (escalated in a prior session). Saw the same four test-only-caller warnings
from sync_derived_docs.py again — no further action needed (fragment is in todo.d for owner).

**sync_derived_docs.py**: ran; "some issues auto-fixed" with zero git diff = internal state
flush only (known pattern — trust git diff).

**`tidy_advisory.json` guard**: this file was dirty BEFORE the run started (visible in initial
git status). Was NOT committed — only the doc-sync changes were staged.

**Fragile cross-references to watch**:
- `completed.d/2026-08-07_lensing-training-path-per-region.md` is now linked from both
  `lensing_chart_kinds_should_share_one_tiling_machine.md` and
  `lensing_wedge_probe_charts_need_retraining_under_v3.md`. If the completed file is renamed,
  both links dangle.
- `guard_slow_operation` is cited nowhere in docs — the function is self-documented via
  its docstring. If it's ever renamed, no doc update is needed, but the surrogate_training
  module docstring should be checked.
