# Librarian Short-Term Observations

## 2026-08-11 -- post-commit sync for b64480c / b9a9ee5 (NO-OP)

Scope: cusp-arm routing fix + interior-cusp-serving-barrier TODO/brief.

### Commits triaged

**b64480c** (feat(lensing): cusp-arm routing fix):
- Changed `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py` and
  `cogwheel/tests/test_lensing_airy_fold.py`.
- Only change is to `_cusp_vertex` — a private function (underscore prefix).
  Implementation changed from seed_theta-nearest-cusp selection to
  source-plane-distance-nearest across all candidate cusps. Public
  behavior (Pearcey arm serves sources within `_CUSP_ARM_COVERAGE` of cusp
  vertex) is unchanged; only the WHICH vertex is selected was fixed.
- SPEC.md does not describe the internal `_cusp_vertex` selection mechanism —
  it only documents the coverage window and the arm's serve contract.
  No staleness. Test file is test-only (per the post-commit NO-OP rule).
- NO doc surface changes required.

**b9a9ee5** (docs: todo + brief for interior cusp serving barrier build):
- Added `.claude/handoff/brief_interior_cusp_serving_barrier.md` (agent-only).
- Added `.claude/spec/todo.d/lensing_interior_cusp_serving_barrier.md`.
- TODO.md was regenerated as part of the commit (render_fragments.py ran).
  Verified: "Interior cusp sources still refuse" appears once in TODO.md.
- No cogwheel/ Python changes. NO-OP.

**572b452** — prior no-op sync commit, not stale.

### sync_derived_docs.py

Ran cleanly. The recurring `lens_amplification_surrogate` test-only consumer
warnings appeared again (same 4 warnings from test_lensing_surrogate.py).
The escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md`
is already open — did NOT create a duplicate. "Auto-fixed" was the known
false-positive state flush; `git diff` confirmed no doc surface changes.

### Pattern note

Private-function implementation fixes (underscore-prefixed, no API
surface change, no new disk artifact) are librarian no-ops even when the
diff is substantial. Only check SPEC.md if the function is NAMED there or
the fix changes a documented constant/coverage window.
