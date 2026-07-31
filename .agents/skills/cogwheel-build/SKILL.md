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

Use `.claude/sdk/launch_build.sh` — the ONLY sanctioned build launcher for all
providers. It handles conda resolution, watchdog attachment, `--approval-dir`
defaulting, depth guards, `disown`, and provider routing automatically.

Builds are long-running — detach them so they survive tool-call timeouts:

- **Claude Code**: Use `Bash` with `run_in_background: true`:
  `.claude/sdk/launch_build.sh <slug> .claude/handoff/<slug>.md`
- **OpenCode/Codex**: Use `nohup ... & disown`:

```bash
# OpenCode (pass OPENCODE_SESSION_ID so the resume callback can wake you):
nohup bash -c 'AGENT_PROVIDER=opencode OPENCODE_SESSION_ID=<this_session> .claude/sdk/launch_build.sh <slug> .claude/handoff/<slug>.md' > /tmp/<slug>_stdout.log 2>&1 &
disown

# Codex (CODEX_THREAD_ID is ambient in the Codex shell — just pass it through):
nohup bash -c 'AGENT_PROVIDER=codex .claude/sdk/launch_build.sh <slug> .claude/handoff/<slug>.md' > /tmp/<slug>_stdout.log 2>&1 &
disown
```

**Note on callbacks:** Codex automatically has `CODEX_THREAD_ID` in its shell
env — it flows through `nohup bash -c` naturally. OpenCode does NOT have its
session ID in the env, so you must pass `OPENCODE_SESSION_ID` explicitly
(get it from `opencode session list` or the session header). Without it, the
build still runs but cannot notify you on completion/escalation.

**Do NOT pass `--auto`** to `launch_build.sh` unless the user explicitly says
to auto-approve. The default is interactive plan review via `--approval-dir`.

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

How to know when the build needs attention (plan approval, terminal event):

- **Claude Code**: Use the native Monitor tool to watch for `plan_ready` or
  terminal markers in the log. Event-driven — no polling.
- **Codex**: The `CODEX_THREAD_ID` resume callback injects a message into
  this thread on plan_ready, escalation, and terminal.
- **OpenCode**: The `OPENCODE_SESSION_ID` resume callback injects a message
  via the serve API (POST /session/<id>/message) on plan_ready, escalation,
  and terminal. Requires `opencode serve` running with
  `OPENCODE_SERVER_PASSWORD` set. Fully autonomous — no polling needed.

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