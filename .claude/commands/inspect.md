Launch the Inspector as a subagent.

1. Read the agent prompt from `.claude/crew/inspector.md`
2. Read the agent state from `.claude/agent_state/inspector.json`
3. Run `git diff $(jq -r .last_commit .claude/agent_state/inspector.json)..HEAD` to get all changes since the last Inspector run
4. Call the Agent tool with:
   - The full agent prompt as the main prompt
   - Append the state JSON under a `## Current State` header
   - Append the scoped git diff output under a `## Changes Since Last Review` header
   - Append user arguments under a `## Arguments` header: $ARGUMENTS
5. Present the agent's complete output to the user without summarizing or filtering
6. **MANDATORY**: after presenting, update `.claude/agent_state/inspector.json` so the next Inspector run scopes its diff correctly:
   ```
   python scripts/update_agent_state.py inspector --status PASS
   ```
   (Use `--status ISSUES` instead if the Inspector returned a non-PASS verdict.) Skipping this step strands the next Inspector run on a stale `last_commit` and it will diff the full history since the last SDK-orchestrated build.
