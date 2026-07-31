---
name: cogwheel-build
description: Run or resume the repository's deep multi-role scientific software build pipeline with Codex or OpenCode while preserving the established Claude workflow.
---

# Cogwheel Build

Use this skill when the user asks to launch the agentic build pipeline, run a
deep implementation build, or resume a build driven by the shared SDK.

1. Read `AGENTS.md` and the required knowledge anchors named there.
2. Use `.codex/build "<task>"` for a Codex-backed build. Pass through normal
   pipeline flags such as `--fast`, `--plan-only`, `--yes`, `--log`, and
   `--approval-dir`.
3. Use `.opencode/build "<task>"` for an OpenCode-backed build. Same flags
   apply.
4. Use `.claude/build "<task>"` only when the user explicitly wants the
   Claude-backed pipeline. Its default has intentionally not changed.
5. Treat `.claude/spec/`, `.claude/crew/`, `.claude/handoff/`, and
   `.serena/memories/` as shared state. Do not create provider-specific copies.
6. For a long build (any provider), the build launcher attaches the shared
   watchdog and emits a callback only for escalation or terminal state. Never
   create a persistent goal or poll the log.
7. Never launch a cost-bearing science campaign without the estimate required
   by `AGENTS.md`, and never run the slow test tier inside a build.

## Codex Backend

The Codex adapter uses `codex exec --json`; authentication, hooks, and thread
persistence remain Codex-native. The long-lived orchestrator starts one
build-scoped Serena Streamable HTTP server on `CODEX_SERENA_PORT` (default
`8324`) and points every Codex role at that warm server; it does not repeatedly
start the interactive stdio configuration. `CODEX_MODEL` and
`CODEX_REASONING_EFFORT` provide global overrides; role-specific forms such as
`CODEX_MODEL_TEST_DEV` take precedence. Without overrides, the Architect and
planning Professor use `gpt-5.6-sol` at high effort; execution/review roles use
`gpt-5.6-terra` at high effort and administrative support roles use it at medium
effort. The adapter raises asyncio's newline reader limit to 8 MiB because one
`codex exec --json` tool event can contain a large whole-file result;
`CODEX_JSON_STREAM_LIMIT` is the escape hatch.

## OpenCode Backend

The OpenCode adapter uses `opencode run --format json`; session persistence,
plugins, and MCP servers remain OpenCode-native. The long-lived orchestrator
starts one build-scoped Serena Streamable HTTP server on `OPENCODE_SERENA_PORT`
(default `8325`) and points every OpenCode role at that warm server via the
`OPENCODE_SERENA_URL` environment variable. `OPENCODE_MODEL` and
`OPENCODE_VARIANT` provide global overrides; role-specific forms such as
`OPENCODE_MODEL_TEST_DEV` take precedence. Without overrides, the Architect,
Coder, Inspector, Professor, and ProfReview use `claude-v4.6-opus` at high
variant; other roles use `claude-v4.6-sonnet`. The adapter raises asyncio's
newline reader limit to 8 MiB; `OPENCODE_JSON_STREAM_LIMIT` is the escape hatch.
