# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for b80d1d6

**Scope**: commit `b80d1d6` — "feat(lensing): serve on-axis interior cusp sources via the Pearcey arm"

**Changed files**: `_airy_fold.py`, `_pearcey_cusp.py`, `operator.py`, tests (3), TODO.md, new todo.d fragment

**What was stale and why**:
- `SPEC.md` "INTERIOR CUSP SERVING" sentence said "interior sources (3 real stationary points, `rho < 1`)" — stale because the bypass in `_pearcey_cusp.cusp_amplification` was extended to also cover the interior degenerate cluster (on-axis interior sources where the first-order control projection degenerates to 1 stationary point but `len(images) > 2`).
- The TODO fragment `lensing_fold_pair_drops_third_cusp_image.md` was added in the SAME commit that implements the fix (unusual pattern: TODO opened and fixed together). Its acceptance criteria (second OR condition: "fold arm correctly detects the 3-image cluster and declines to the cusp arm") was met by the commit. Closed it.

**Pattern noted**: A TODO fragment opened AND fixed in the same commit is a valid pattern here — the driver documents the problem history then immediately applies the fix. The fragment still lands in todo.d (not completed.d) in that commit, leaving the closure to the Librarian. Watch for this pattern in future post-commit syncs.

**operator.py change**: SERVING LADDER ordering (uniform arm before Schwinger exact) was already correct in SPEC; the operator.py bug was that the mpmath band routed directly to exact engine without offering the uniform arm first. No SPEC update needed for this — SPEC described correct architecture, code had the bug.

**Fragments created**:
- `.claude/spec/spec_changelog.d/2026-08-12_on_axis_interior_cusp_bypass.md` (patch bump)
- `.claude/spec/completed.d/2026-08-12_on_axis_interior_cusp_bypass.md`

**Fragment deleted**: `.claude/spec/todo.d/lensing_fold_pair_drops_third_cusp_image.md`

**Fragile cross-refs added**: `_CUSP_TIE_EPS = 1e-12` and `_merging_fold_pair` now cited in SPEC.md INTERIOR CUSP SERVING — if either is renamed or the constant changes, SPEC goes stale.

**Skipped**: test-only changes in 3 test files (skip per POST-COMMIT SYNC NO-OP RULE for test-only). No new disk artifacts in this commit (all in-memory computation changes). DATA_CONTRACTS.yaml not touched.

**sync_derived_docs.py output**: only the pre-existing `lens_amplification_surrogate` test-consumer warning (already escalated via todo fragment `surrogate_contract_test_consumer_warning.md`). No new issues.
