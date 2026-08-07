# Librarian Short-Term Observations

## 2026-08-07 post-commit sync (commit 3aedd41)

**Scope**: scripts: add progress beats to driver probes (observable runs).

**Changed files**: `scripts/probe_exterior_recursion.py`, `scripts/probe_wedge_v3.py` — driver
probe scripts only; no `cogwheel/` library code touched.

**Result**: Confirmed no-op per SCRIPTS/ REWRITE NO-OP RULE. These scripts stay under
`scripts/`, introduce no new serialization artifacts, and make no changes to the
`cogwheel/` public API. All doc surfaces (SPEC.md, DATA_CONTRACTS.yaml, overview.rst,
api.rst, crash_course.rst) remain accurate with no edits needed.

**sync_derived_docs.py**: same four test-only-caller consumer-graph warnings for
`lens_amplification_surrogate` — pre-existing, already escalated via
`surrogate_contract_test_consumer_warning.md` todo fragment. No new action.
"Some issues auto-fixed" with only stray diffs in `tidy_advisory.json` and
`librarian.json` — reverted both (known side effect, not real doc changes).

**Pattern noted**: scripts-only commits (progress beats, probe refactors) are routine
no-ops for the librarian. The SCRIPTS/ REWRITE NO-OP RULE covers even large additions
(100+ lines) when the script stays in `scripts/` and has no serialization artifacts.

**Fragile cross-references to watch** (carried forward from prior session):
- `completed.d/2026-08-07_lensing-training-path-per-region.md` is linked from
  `lensing_chart_kinds_should_share_one_tiling_machine.md` and
  `lensing_wedge_probe_charts_need_retraining_under_v3.md`. If the completed file is
  renamed, both links dangle.
- `guard_slow_operation` cited nowhere in docs — self-documented via docstring.
