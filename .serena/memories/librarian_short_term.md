# Librarian Short-Term Observations

## 2026-08-19 (tiling_plan doc sync, order-7a step 2)

- Scope: build shipped `cogwheel/lensing/tiling_plan.py` + CLI `scripts/tiling_plan.py`
  + `cogwheel/tests/test_lensing_tiling_plan.py` (engine-free demand-sized tiling
  plan/cost predictor, order-7a step 2 of `todo.d/lensing_training_campaign`) plus
  an in-build fix for Inspector finding INS-1-001 (DD-band w-axis ceiling clip via
  new `_resolve_dd_ceiling`). Files-changed list showed ZERO `.claude/spec/*`
  touches despite a new production module + CLI shipping — same "new lensing
  module lands with no accompanying spec/changelog/completed fragment" pattern
  noted for tiling_census (08-14) and serve_route_census (08-17); now a 5th+
  confirmation this is the STANDING state after every lensing-census-family
  build, not an anomaly to keep re-discovering.
- Fixes applied: (1) inserted a new dense paragraph into SPEC.md's Microlensing-
  engine table row (line 53) documenting the module's mechanism, the INS-1-001
  fix, the three reported (non-fatal) cross-checks, the escalation verdict, and
  explicitly that this is PLANNING only (order-7a step 2 done, steps 3-4 still
  open) — via `mcp__serena__replace_content` literal mode, verified with
  `search_for_pattern` before AND diff-stat after (3 del/3 ins: 2 header lines +
  1 paragraph line, matches expectation exactly, no corruption). (2) Created
  `spec_changelog.d/2026-08-19_tiling_plan.md` (bump: minor) and
  `completed.d/2026-08-19_tiling_plan.md` (section: Lensing training), the
  latter explicitly stating `todo.d/lensing_training_campaign` is NOT closed by
  this entry (steps 3-4 remain open), matching the 2026-08-17 serve_route_census
  precedent's phrasing pattern exactly. (3) Ran render_fragments.py -> spec_version
  bumped 0.48.0 -> 0.49.0, SPEC_CHANGELOG.md/COMPLETED.md regenerated cleanly.
- Skipped with reason: DATA_CONTRACTS.yaml — no edit needed; tiling_plan.py's own
  JSON output lives in `.claude/handoff/` (advisory/debug artifact class, same as
  tiling_census/serve_route_census's own outputs, historically untracked) and it
  never attaches the `lens_amplification_surrogate` artifact (always calls
  `serve_route_census.run(..., artifact=None)`, demand-mode only). Grepped
  DATA_CONTRACTS.yaml for "handoff" — zero matches, confirming this class of
  output is consistently out of contract scope. docs/source/{api,overview,
  crash_course}.rst — no edit needed: api.rst's `:recursive:` autosummary on the
  single `cogwheel` top-level entry auto-covers new submodules (satisfies
  Enforcement rule 6 without a manual edit); overview.rst/crash_course.rst
  correctly exclude internal build/planning tooling from user-facing narrative
  (5th confirmation of this standing rule — see knowledge memory). No Sphinx
  rebuild performed since nothing under docs/source/ was touched.
- Side-effect discipline: `render_fragments.py` again dirtied
  `.claude/tidy_advisory.json` as a stray side effect (7 ins/3 del) with no
  relation to this fragment — reverted via `git checkout --` per standing
  knowledge-memory rule. `foreman_lite.json` was NOT touched this run (unlike
  some prior sessions) — the side-effect file set can vary run to run, always
  check `git status --porcelain` after render rather than assuming a fixed pair.
  Left untouched (out of Librarian scope, pre-existing/other-agent dirty state,
  not caused by this run): `.claude/agent_state/{architect,coder,inspector,
  test_dev,tidy}.json` and `.serena/memories/{architect,coder,inspector,
  professor,test_dev}_short_term.md` — all showed as modified in git status
  before I touched anything; these belong to other agent roles, not mine to
  revert or stage.
- Fragile cross-reference to watch: SPEC.md's new tiling_plan paragraph cites
  `_CENSUS_ENGINE_RESIDUAL_LEDGER = 0.4119` (the `campaign_tiling_design` Fact-1
  honest-ledger constant) by literal value — if that Fact 1 or the constant is
  ever remeasured/changed in code, this SPEC sentence needs a matching update.
  Also cites the two new DD-ceiling-clip source tags (`measured_clipped_dd`,
  `prior_box_fallback_clipped_dd`) by exact string — renaming either breaks the
  cross-ref silently (same family as the schema-constant-name fragility already
  tracked in knowledge memory).
- Pre-existing dangling-wiki-link warnings from render_fragments.py (6 of them,
  e.g. `[[FINDINGS F069/F070/F071/F072]]`, `[[lensing_born_farfield_completion]]`)
  are unrelated to this fragment and were already present before this run —
  confirmed by their filenames not overlapping with anything touched here. Left
  alone; this is the known "check_wiki_links only resolves todo.d/completed.d
  stems, never taught the FINDINGS-bracket convention" tooling gap already
  filed as a todo.d fragment per knowledge memory (not re-filing a duplicate).
