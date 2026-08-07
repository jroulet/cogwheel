# Librarian Short-Term Observations

## 2026-08-07 post-commit sync (commit e3fb557)

**Scope**: Single commit — fix(opencode): correct Go model format + inherit interactive subagent models.

**Changed files**: `.claude/sdk/runtime_opencode.py` (minor format fix), `.env.example` (+7 lines),
`.opencode/agents/*.md` (11 agent frontmatter files: removed `model:`/`variant:` lines).

**Stale surface found and fixed**: AGENTS.md (= CLAUDE.md symlink).
- `test_dev` was listed in the sonnet-tier group ("Foreman-Lite, Test Developer, Librarian...") but
  `runtime_opencode.py` has always assigned it to opus-tier. Pre-existing gap; fixed now.
- `OPENCODE_MODEL_PROVIDER` is a new env var (documented in `.env.example`) that selects between
  AI Commons (default) and OpenCode Go native models (deepseek-v4-pro/flash). Was not mentioned
  in AGENTS.md; added in this sync.

**What went stale and why**: AGENTS.md wasn't touched in either the big refactor commit (1ffaea3)
or the fix commit (e3fb557). The env var addition pattern: new env vars documented in .env.example
don't auto-propagate to AGENTS.md — requires manual sync each time.

**Fragile cross-references to watch**:
- AGENTS.md now names `opencode-go/deepseek-v4-pro` and `opencode-go/deepseek-v4-flash` explicitly —
  if these model strings change in runtime_opencode.py, AGENTS.md needs updating.
- If AI Commons model names change (claude-v4.6-opus → something else), AGENTS.md and .env.example
  both need updating simultaneously.
- The `.opencode/agents/*.md` frontmatter now carries NO `model:` line by design. If this convention
  changes back, AGENTS.md would need a new note.

**SCRIPTS/ REWRITE NO-OP**: Both runtime_opencode.py and the .opencode/agents/ files are agent-infra
paths. The Sphinx doc surfaces (overview.rst, api.rst, installation.rst, crash_course.rst) are
unaffected — no cogwheel/ code changed.

**sync_derived_docs.py**: Not run (no cogwheel/ changes to check).
**render_fragments.py**: Not run (no fragments written).
