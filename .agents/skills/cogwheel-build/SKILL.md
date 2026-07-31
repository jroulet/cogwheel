---
name: cogwheel-build
description: Run or resume the repository's deep multi-role scientific software build pipeline with Codex or OpenCode while preserving the established Claude workflow.
---

# Cogwheel Build

Use this skill when the user asks to launch the agentic build pipeline, run a
deep implementation build, or resume a build driven by the shared SDK.

## Prerequisites

1. Read `AGENTS.md` and the required knowledge anchors named there.
2. Never launch a cost-bearing science campaign without the estimate required
   by `AGENTS.md`, and never run the slow test tier inside a build.

## Launching a build

Use the provider-appropriate launcher. Builds are long-running and MUST be
detached from the interactive shell so they survive tool-call timeouts:

- **Claude Code**: Use `Bash` with `run_in_background: true` (native feature).
- **OpenCode/Codex**: Wrap in `screen -dmS <name> bash -c '...'` so the build
  runs in a detached screen session. Do NOT call the build script directly from
  a bash tool call — the tool timeout will kill it.

```bash
# OpenCode example:
screen -dmS build_1b bash -c 'OPENCODE_SESSION_ID=<session> .opencode/build "@.claude/handoff/<brief>.md" --approval-dir /tmp/<slug>_approval > /tmp/<slug>_stdout.log 2>&1; echo "EXIT: $?" >> /tmp/<slug>_stdout.log'

# Codex example:
screen -dmS build_1b bash -c 'CODEX_THREAD_ID=<thread> .codex/build "@.claude/handoff/<brief>.md" --approval-dir /tmp/<slug>_approval > /tmp/<slug>_stdout.log 2>&1; echo "EXIT: $?" >> /tmp/<slug>_stdout.log'
```

- `.claude/build "<task>"` only when the user explicitly wants the Claude pipeline.

**Critical flags:**
- `--approval-dir <dir>` — REQUIRED for interactive plan review. The build
  blocks until you approve or reject.
- `--fast` — skip Phase 1 planning (trivial tasks only).
- `--plan-only` — plan + approval then stop (dry run).
- `--log <file>` — explicit log path (auto-generated if omitted).
- `OPENCODE_SESSION_ID=<this_session>` or `CODEX_THREAD_ID=<this_thread>` —
  set so terminal/escalation callbacks resume this session automatically.

**Do NOT pass `--yes`** unless the user explicitly says to auto-approve.
The default is interactive plan review.

## Plan review workflow

1. Launch the build with `--approval-dir`.
2. Monitor the log for `Phase 1: Planning` and `Plan written`.
3. When `<approval-dir>/plan_ready` appears, read `<approval-dir>/plan.json`.
4. Evaluate the plan:
   - Are work packages well-scoped?
   - Do file lists make sense?
   - Are max_turns estimates reasonable?
5. Decide:
   - **Approve**: `touch <approval-dir>/plan_approved`
   - **Reject**: `echo "feedback" > <approval-dir>/plan_rejected`
     (the Architect revises with your feedback)
   - **Escalate to user**: if genuinely unsure about correctness.
6. Phase 2 runs autonomously after approval. Monitor via the log.
7. On terminal (success/failure), the resume callback fires and you assess.

## Monitoring

The build prints a Monitor command in its log header. Use it or tail the log.
Health = log mtime advancing, NOT pgrep. The watchdog kills stalled builds at
the configured threshold (default 1200s).

## Codex Backend

The Codex adapter uses `codex exec --json`. The orchestrator starts one shared
Serena Streamable HTTP server on `CODEX_SERENA_PORT` (default `8324`).
`CODEX_MODEL` / `CODEX_REASONING_EFFORT` provide global overrides; role-specific
forms like `CODEX_MODEL_TEST_DEV` take precedence.

## OpenCode Backend

The OpenCode adapter uses `opencode run --format json`. The orchestrator starts
one shared Serena Streamable HTTP server on `OPENCODE_SERENA_PORT` (default
`8325`). `OPENCODE_MODEL` / `OPENCODE_VARIANT` provide global overrides;
role-specific forms like `OPENCODE_MODEL_TEST_DEV` take precedence.

## Shared state

Treat `.claude/spec/`, `.claude/crew/`, `.claude/handoff/`, and
`.serena/memories/` as shared state. Do not create provider-specific copies.