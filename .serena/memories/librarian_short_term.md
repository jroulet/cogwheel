# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for 16aacc0

**Scope**: commit `16aacc0` — "feat(lensing): serve saddle deltoid interior cusp sources + fix stale refusal fixtures"

**Changed files (non-test)**: `_pearcey_cusp.py`, `.claude/spec/TODO.md`, `.claude/spec/todo.d/lensing_saddle_origin_rho_assumption.md`

**What was stale and why**:
- SPEC.md INTERIOR CUSP SERVING section described the interior discriminator as `rho < 1` (origin-based directional `r_caustic` check). The code changed to `_is_interior = len(images) >= 4` — the parity-correct discriminator for both astroid and deltoid. The deltoid caustic does not enclose the origin, so `r_caustic` raised `LensDomainError` for saddle corridor rays and defaulted to `_is_interior = False`, blocking deltoid interior cusp serving. The new discriminator enables it.
- Stale references updated: "(1) generic interior case (3 real stationary points, `rho < 1`)" → "`len(stationary_values) == 3`"; "(2) interior degenerate cluster (`rho < 1`, `len(images) > 2`, ...)" → "`len(images) >= 4`, `len(stationary_values) == 1`"; "Exterior sources (1 stationary point, `rho > 1`)" → "`len(images) < 4`".

**New TODO fragment**: `lensing_saddle_origin_rho_assumption.md` — open bug in ppGO/Born interior handoff (likelihood.py:1396/1681) that misclassifies saddle corridor sources. No doc surface update needed for a newly-opened TODO.

**Fragments created**:
- `.claude/spec/spec_changelog.d/2026-08-12_saddle_deltoid_interior_cusp_image_count.md` (patch bump)

**render_fragments.py output**: updated SPEC_CHANGELOG.md and SPEC.md spec_version → 0.37.10. Also re-rendered TODO.md (20 lines added — likely formatting/ordering update from the todo.d fragment landing in the canonical). Side effects to watch: agent_state/*.json and tidy_advisory.json also show dirty from render_fragments.py; caller should stage only the spec files.

**Skipped**: all 6 test file changes (test-only per triage table). No new disk artifacts in this commit. DATA_CONTRACTS.yaml not touched.

**Fragile cross-refs in INTERIOR CUSP SERVING**: `_is_interior = len(images) >= 4` and `_merging_fold_pair` and `_CUSP_TIE_EPS = 1e-12` all cited in SPEC — if any rename, SPEC goes stale.

**Pattern noted**: the `lensing_saddle_origin_rho_assumption` TODO is an open production bug in likelihood.py that spans multiple sites (1396, 1681, surrogate_census.py, surrogate_training.py). When a future build fixes it, expect SPEC.md's "FOLD-PPGO INTERIOR HANDOFF" section (which currently says `rho <= 1.0` for positive parity interior) and DATA_CONTRACTS.yaml consumer entries to need simultaneous updates.

**sync_derived_docs.py**: only the pre-existing `lens_amplification_surrogate` test-consumer warning (already escalated via open todo fragment `surrogate_contract_test_consumer_warning.md`). No new issues.
