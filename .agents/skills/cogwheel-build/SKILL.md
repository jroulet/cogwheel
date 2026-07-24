---
name: cogwheel-build
description: Run or resume the repository's deep multi-role scientific software build pipeline with Codex while preserving the established Claude workflow.
---

# Cogwheel Build

Use this skill when the user asks to launch the agentic build pipeline, run a
deep implementation build, or resume a build driven by the shared SDK.

1. Read `AGENTS.md` and the required knowledge anchors named there.
2. Use `.codex/build "<task>"` for a Codex-backed build. Pass through normal
   pipeline flags such as `--fast`, `--plan-only`, `--yes`, `--log`, and
   `--approval-dir`.
3. Use `.claude/build "<task>"` only when the user explicitly wants the
   Claude-backed pipeline. Its default has intentionally not changed.
4. Treat `.claude/spec/`, `.claude/crew/`, `.claude/handoff/`, and
   `.serena/memories/` as shared state. Do not create a parallel Codex copy.
5. For a long or detached build, follow the monitor command printed by the
   launcher and keep monitoring until a terminal state. Do not infer health
   from the wrapper process alone.
6. Never launch a cost-bearing science campaign without the estimate required
   by `AGENTS.md`, and never run the slow test tier inside a build.

The Codex adapter uses `codex exec --json`; authentication, hooks, and thread
persistence remain Codex-native. The long-lived orchestrator starts one
build-scoped Serena Streamable HTTP server on `CODEX_SERENA_PORT` (default
`8324`) and points every Codex role at that warm server; it does not repeatedly
start the interactive stdio configuration. `CODEX_MODEL` and
`CODEX_REASONING_EFFORT`
provide global overrides; role-specific forms such as
`CODEX_MODEL_TEST_DEV` take precedence. Without overrides, scientific
authority roles use `gpt-5.6-sol` at high effort and bounded support roles use
`gpt-5.6-terra` at medium effort. The adapter raises asyncio's newline reader
limit to 8 MiB because one `codex exec --json` tool event can contain a large
whole-file result; `CODEX_JSON_STREAM_LIMIT` is the escape hatch.
