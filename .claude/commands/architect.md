This command has been superseded by `/build`, which runs the SDK pipeline
with in-session plan review and graceful degradation.

Use `/build` instead. It launches the full SDK orchestrator in background,
lets you (the session agent) review the Architect's plan via file-based
signaling, and recovers from crashes by finishing remaining agents manually.

The Architect crew prompt (`.claude/crew/architect.md`) is still used by the
SDK pipeline — only this command (the in-session orchestration wrapper) is retired.
