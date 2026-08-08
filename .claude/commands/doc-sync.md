Launch the Librarian as a subagent to sync documentation with recent code changes.

1. Read the crew prompt from `.claude/crew/librarian.md`
2. Call the Agent tool with:
   - Do not pass an explicit `model` — the Librarian agent frontmatter
     (`.opencode/agents/librarian.md`) carries the provider-correct model
   - The full crew prompt as the system context
   - Append the current `git log --oneline -10` and `git diff --stat HEAD~3` under a `## Recent Changes` header
   - Append user arguments under a `## Task` header: $ARGUMENTS
   - If no arguments provided, audit all doc surfaces for staleness
3. Present the agent's complete output to the user without summarizing or filtering
4. **MANDATORY**: after presenting, update `.claude/agent_state/librarian.json`:
   ```
   python scripts/update_agent_state.py librarian
   ```
   Skipping this step strands the next Librarian run on a stale `last_commit`.

## Post-commit mode

When called with `--post-commit <sha>` (by the post-commit hook or the session
agent), the Librarian runs in autonomous mode: reads `.claude/sync_issues.json`
for context, fixes stale doc surfaces, commits its own fixes, and cleans up the
issues file. See the "Post-commit mode" section in `.claude/crew/librarian.md`
for full behavioral differences.

## Trigger file

After every doc-relevant commit, the post-commit hook writes
`.claude/sync_issues.json` with the commit SHA and changed files. When inside
a Claude Code session, the hook does NOT launch a new claude process (nesting
crashes). Instead, the session agent should check for the trigger file and
run `/doc-sync --post-commit <sha>` itself after completing its current work.

Check: `test -f .claude/sync_issues.json && echo "Librarian sync pending"`
