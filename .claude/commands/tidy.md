Launch the Tidier as a subagent.

1. Read the agent prompt from `.claude/crew/tidy.md`
2. Read the agent state from `.claude/agent_state/tidy.json`
3. Call the Agent tool with:
   - The full agent prompt as the main prompt
   - Append the state JSON under a `## Current State` header
   - Append user arguments under a `## Arguments` header: $ARGUMENTS
4. Present the agent's complete output to the user without summarizing or filtering
5. **MANDATORY**: after presenting, update `.claude/agent_state/tidy.json`:
   ```
   python scripts/update_agent_state.py tidy
   ```
