Launch the SDK pipeline orchestrator from within this session.

You (the session agent) take the user's role: launch the build, review the
Architect's plan, approve or reject, monitor execution, and recover from crashes.
The Python orchestrator retains all deterministic guarantees — you don't replace
the Inspector, Tidier, Test Dev, or Librarian. You intervene only on plan review
and failure recovery.

## Argument parsing

`$ARGUMENTS` may contain CLI flags before or after the task description.
Extract these known flags and forward them to `build.py`:

| Flag | Effect |
|------|--------|
| `--fast` | Skip Phase 1 planning; Foreman-Lite handles directly |
| `--plan-only` | Run Phase 1 only (plan + approval), then stop |
| `-v` / `--verbose` | Show agent text output (thinking, commentary) |
| `-q` / `--quiet` | Phase transitions + final report only |
| `--no-serena` | Disable Serena MCP (use built-in tools only) |

Everything remaining after flag extraction is the task description (`$TASK`).
Do NOT forward `--yes`, `--approval-dir`, `--serena-url`, or `--log` — those are
managed by this command.

If `--no-serena` is set, omit `--serena-url` from the invocation entirely.

## Prerequisites

1. Verify branch safety (if ALLOWED_BRANCHES is configured in gates.py)

## Execution flow

### Phase 1: Launch and plan review

2. Create the build directory:
   ```bash
   mkdir -p .claude/build
   ```

3. Launch the SDK build + watchdog sidecar in background:
   ```bash
   LOG=.claude/sdk/logs/build_$(date +%Y%m%d_%H%M%S).log
   .claude/sdk/watchdog.sh "$LOG" 600 &
   python .claude/sdk/build.py build \
       --approval-dir .claude/build/ \
       --log "$LOG" \
       $EXTRA_FLAGS \
       "$TASK"
   ```
   Where `$EXTRA_FLAGS` is the extracted flags (e.g. `--fast -v`).
   Use the Bash tool with `run_in_background: true`.

   **Note on `--serena-url`**: omit by default. The SDK spawns its own
   Serena SSE server on port 8322 — no pre-existing Serena required. If
   the user tells you they have a warm SSE Serena already running (e.g.
   manually launched to share a pyright cache across back-to-back builds),
   append `--serena-url http://localhost:8322/sse`. The SDK probes the
   URL at startup and falls back to spawning its own if the probe fails,
   so passing the flag when nothing's there is safe (just slightly slower).

   The watchdog polls the log's mtime every 30s and kills the orchestrator
   subtree if the log goes stale for >600s (10 min — the wedge threshold).
   It self-terminates when the build exits, so the shell's completion
   notification fires on build end regardless of exit path (clean / failure
   / watchdog-kill).

   **If `--fast` is set**: skip steps 4-6 entirely — there is no plan to review.
   Go straight to Phase 2 (monitoring).

   **If `--plan-only` is set**: after step 6 (plan approval), the build exits
   automatically. Go straight to Phase 3 (report).

4. Wait for the plan: use the Monitor tool to watch for `.claude/build/plan_ready`
   with a timeout of ~30 min. Architect typically produces the plan in 2-5 min;
   Monitor fires as soon as the file appears.

   Do NOT poll with a Bash `until/sleep` loop — the `use-serena.sh` hook blocks
   `test` (not in its read-only allowlist), and Serena's `execute_shell_command`
   isn't designed for long-running loops (it will time out).

5. When ready, read `.claude/build/plan.json` and evaluate:
   - Are the work packages well-scoped?
   - Do the file lists make sense for the task?
   - Are max_turns estimates reasonable?

6. Decide:
   - **Approve**: `touch .claude/build/plan_approved`
   - **Reject with feedback**: `echo "feedback text" > .claude/build/plan_rejected`
     (the SDK resumes the Architect with your feedback; a new `plan_ready` will appear)
   - **Escalate to user**: Only if you are genuinely unsure about the plan's
     correctness. Use AskUserQuestion to present the plan and get the user's decision.

### Phase 2: Monitor execution

7. The SDK runs Phase 2 deterministically (Coder → Tidier → TestDev → Inspector → Librarian).
   No input needed from you. Monitor progress by tailing the log if desired.

8. Wait for the background build to complete (you'll be notified automatically).

### Phase 3: Report and cleanup

9. Read the build log tail to get the final report.

10. Report results to the user concisely:
    - Work packages completed
    - Inspector verdict
    - Commits made
    - Any escalations or failures

11. Clean up:
    ```bash
    rm -rf .claude/build/
    ```

### Crash recovery

If the build fails or is killed mid-execution:

12. Read the log to determine where it stopped.
13. Read `.claude/agent_state/*.json` to see which agents completed.
14. Read `.claude/build/plan.json` for the plan context.
15. Finish the remaining work:
    - If code changes are incomplete: complete them directly or describe what's left
    - If Inspector hasn't run: `/check`
    - If Tidier hasn't run: `/tidy`
    - If Librarian hasn't run: `/doc-sync`
    - If Dreamer hasn't run: `/dream`
16. Report to the user what happened and what you recovered.
17. Clean up: `rm -rf .claude/build/`

## Important

- The build runs with the session's Serena instance (--serena-url). No port conflict.
- You are the plan reviewer. Auto-approve unless the plan looks wrong.
- Only escalate to the user if something is genuinely unexpected or ambiguous.
- The SDK handles all commits, Inspector verification, doc sync, and memory consolidation.
  You only intervene on failure.
- If $ARGUMENTS is empty (no flags and no task), ask the user what they want to build.
