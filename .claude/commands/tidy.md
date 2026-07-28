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
   Skipping this step leaves `tidy.json` reading `status: "failed"` from the
   PREVIOUS run, so a Tidier that worked is indistinguishable from one that
   never ran. That is exactly what happened between 2026-07-19 and
   2026-07-28: the role ran and produced real work on 07-27 (a 153-line
   reflow of `cogwheel/lensing/**`), but the state file was never updated and
   the advisory kept accumulating for eight days with nobody the wiser.

   Run it even when the Tidier could not commit — a collision with a live
   build is a normal outcome, and the state still needs to record that the
   run happened. The post-commit hook now prints a loud STALE banner when
   this is skipped.
