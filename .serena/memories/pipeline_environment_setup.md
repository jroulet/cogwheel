# Pipeline Environment Setup (migrated from Claude auto-memory, 2026-07-18)

Source: Claude Code auto-memory `teja-force-pipeline-setup` (originally written
2026-06-05; `type: project`), migrated here so SDK agents can access it directly
without depending on the per-machine `~/.claude/projects/.../memory/` path.

- MACHINE-NEUTRAL NOTE (2026-07-18): this pipeline now runs on more than
  one machine (laptop `/Users/tejaswi/Work/cogwheel-claude-dev`, IAS server
  `/home/tejaswi/Work/cogwheel-claude-dev`). Resolve the worktree from cwd
  and the interpreter as `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`
  — do NOT hard-code either machine's absolute paths. The conda env name is
  routed through the durable `.env` idiom (mirrors gw_detection_ias): set
  `SDK_CONDA_ENV` in an untracked `.env` at the repo root (copy the tracked
  `.env.example`); precedence is shell env > `.env` > default `cogwheel_310`.
  `.claude/sdk/launch_build.sh` and `.claude/build` source it; the launcher
  then resolves the env python absolutely and fails loudly if missing.
- The TejaForce agent build pipeline lives in a git worktree named
  `cogwheel-claude-dev` (sibling of the main repo) on
  branch `claude-dev` — `main` stays clean. The worktree has a scoped
  `core.hooksPath=.claude/hooks` and `ALLOWED_BRANCHES=["claude-dev"]` in
  `.claude/sdk/gates.py`; a pre-push guard blocks pushes to main/master.
- Conda env routing: the pipeline orchestrator AND agents run cogwheel code
  in the environment selected by `SDK_CONDA_ENV`. The portable default is
  `cogwheel_310`; this IAS worktree's untracked `.env` currently selects
  **`cogwheel-newlal`** (Python 3.10, has both `cogwheel` and
  `claude-agent-sdk`) — NOT the main research env `cogwheel` (Python 3.9,
  incompatible with `claude-agent-sdk` >=3.10).
- Installed Claude runtime: `claude-agent-sdk` is **0.1.53** in
  the active `cogwheel-newlal` environment (verified 2026-07-24, but still
  not declared as a project dependency).
  The 0.2.x line (e.g. 0.2.119) times out on the SDK<->CLI handshake at the
  first live `query()` ("Control request timeout: initialize"). All local
  conda envs are x86_64 (Rosetta on an arm64 Mac); the Bun "CPU lacks AVX"
  warning is a red herring — the SDK version, not the arch, is the gate.
- Docs: cogwheel uses Sphinx (`docs/source/`, RST, autosummary in
  `api.rst`), built on Read the Docs.
- Provider layout (2026-07-24): `AGENTS.md` is canonical and `CLAUDE.md`
  symlinks to it. Claude keeps `.mcp.json`, `.claude/settings.json`, and
  `.claude/build`; Codex uses `.codex/config.toml`, `.codex/hooks.json`, and
  `.codex/build`. Both backends share `.claude/sdk`, crew contracts, specs,
  handoffs, and these Serena memories.
- Serena build lifecycle (2026-07-24): interactive sessions own independent
  stdio Serena processes. Each multi-agent build owns one warm SSE Serena
  shared by all of its roles. Claude uses `SDK_SERENA_PORT` (this worktree:
  8323); Codex uses distinct `CODEX_SERENA_PORT` (default 8324), so concurrent
  Claude/Codex builds do not bind or watchdog-kill each other's server.
- To run: `.claude/build "task"` for the unchanged Claude default, or
  `.codex/build "task"` for the Codex runtime.
