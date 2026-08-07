# Librarian Short-Term Observations

## 2026-08-07 post-commit sync (commit 58d4f82)

**Scope**: Single commit — feat(opencode): auto-sync agent models via launch_build.sh.

**Changed files**: `.claude/sdk/launch_build.sh` (+7 lines: calls sync script when
AGENT_PROVIDER=opencode), `.opencode/agents/*.md` (11 files: restored `model:` frontmatter),
`scripts/sync_opencode_agents.py` (new 60-line script).

**Stale surface found and fixed**: AGENTS.md — OpenCode routing bullet lacked any mention of
the new auto-sync mechanism. Added three sentences at the end of the bullet noting that
`.opencode/agents/*.md` frontmatter is auto-synced at build launch via
`scripts/sync_opencode_agents.py` (called by `launch_build.sh`) and no manual frontmatter
edits are needed when switching providers.

**What went stale and why**: AGENTS.md covers model routing env vars but says nothing about
the implementation that keeps `.opencode/agents/*.md` in sync. When `launch_build.sh` gains
a new pre-launch action, AGENTS.md isn't automatically updated — requires manual triage.

**Fragile cross-references to watch**:
- `scripts/sync_opencode_agents.py` name cited in AGENTS.md — if the script is renamed,
  update AGENTS.md.
- If `launch_build.sh` ever drops the auto-sync call (AGENT_PROVIDER=opencode path),
  AGENTS.md's "no manual edits needed" claim becomes false.
- The previous run (e3fb557 / f13b14c) removed frontmatter from `.opencode/agents/*.md`
  and updated AGENTS.md to document `OPENCODE_MODEL_PROVIDER`. This run (58d4f82) restored
  frontmatter and added auto-sync — the AGENTS.md paragraph now reflects the full
  three-part picture: routing table, env var overrides, and auto-sync maintenance.

**SCRIPTS/ REWRITE NO-OP**: All changes are agent-infra paths. The Sphinx doc surfaces
(overview.rst, api.rst, installation.rst, crash_course.rst) are unaffected — no cogwheel/
code changed.

**sync_derived_docs.py**: Not run (no cogwheel/ changes to check).
**render_fragments.py**: Not run (no fragments written).

## Earlier run: 2026-08-07 post-commit sync (commit e3fb557)

**Scope**: Single commit — fix(opencode): correct Go model format + inherit interactive subagent models.

**Changed files**: `.claude/sdk/runtime_opencode.py` (minor format fix), `.env.example` (+7 lines),
`.opencode/agents/*.md` (11 agent frontmatter files: removed `model:`/`variant:` lines).

**Stale surface found and fixed**: AGENTS.md.
- `test_dev` was listed in the sonnet-tier group but runtime_opencode.py always assigned it to opus-tier. Fixed.
- `OPENCODE_MODEL_PROVIDER` new env var (documented in `.env.example`) added to AGENTS.md.

**What went stale and why**: AGENTS.md wasn't touched in either the big refactor commit (1ffaea3)
or the fix commit (e3fb557). New env vars in .env.example don't auto-propagate to AGENTS.md.
