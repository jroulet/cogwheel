# Librarian Short-Term Memory

## Last run: 2026-07-16 (post-commit sync, 6 pending commits)

Processed 6 pending commits from `.claude/sync_issues.json`: 1d149dd, c233fae, 8baa0e7, d5e55a4, 8aa96c2, 8f0f9c3.

**Caveat on the trigger context**: the orchestrator handed me `git log -10` / `diff --stat HEAD~3`
and asserted "no cogwheel/ file changed, SPEC.md not edited" for this batch. That was true only
for the *recent* window it looked at — `1d149dd` (oldest of the 6 pending, 15 commits back) is
NOT agent-infra: it added `cogwheel/lensing/chang_refsdal/{_dd,_gauge,geometry}.py` +
`cogwheel/lensing/__init__.py` + tests, and DID edit `.claude/spec/SPEC.md` +
`SPEC_CHANGELOG.md`. Caught this by re-deriving the actual commit list from
`.claude/sync_issues.json` timestamps against `git log`, not by trusting the summary. Worth
remembering: always re-derive the diff window from sync_issues.json's own commit hashes —
the orchestrator's framing narrows to whatever window it happened to look at.

**Outcome despite the caveat: still nothing to fix.**
- SPEC.md's new "Microlensing engine (IN PROGRESS — foundation only)" row already lists exactly
  the 3 files that exist (`_dd.py`, `_gauge.py`, `geometry.py`) — module attribution correct,
  written by the commit's own author per the Spec/TODO workflow (spec_changelog fragment
  already present and rendered).
- `docs/source/api.rst` uses `.. autosummary:: :recursive:` rooted at `cogwheel` — new
  subpackages (`cogwheel.lensing`, `cogwheel.lensing.chang_refsdal`) are auto-discovered.
  Rule 6 (new top-level module -> API coverage) is satisfied with zero edits needed; do not
  hand-add per-module autosummary entries here.
- `overview.rst` / `crash_course.rst`: correctly do NOT mention lensing yet — SPEC.md itself
  marks it foundation-only/untested (geometry.py untested, kernel/operator/channels still
  pending Build 1b). Adding it to user-facing narrative docs now would be premature/invented
  content; wait until it's a real usable pipeline step.
- `TODO.md` already carries the lensing-program fragment from a prior run (build sequence,
  paper pointer) — unchanged, still accurate, no completed.d fragment yet (Build 1b unfinished).
- `DATA_CONTRACTS.yaml`: no lensing entries — correct, no data artifact registered yet.
- The other 5 pending commits (c233fae, 8baa0e7, d5e55a4, 8aa96c2, 8f0f9c3) touch only
  `.claude/**` (orchestrator/agents.py, crew prompts, handoff briefs, SDK self-tests) —
  agent-only paths, out of doc-sync scope, confirmed via their changed_files lists (no
  cogwheel/ or docs/source/ entries).
- `python scripts/sync_derived_docs.py --check` — clean, exit 0.

Backlog cleared (`.claude/sync_issues.json` deleted — it was untracked in git, not committed).

No doc-surface files touched this run; only this memory file changes, committed per the
established precedent (see f7fd53f) of recording even no-op sync runs.
