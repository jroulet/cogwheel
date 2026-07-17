# Pipeline Environment Setup (migrated from Claude auto-memory, 2026-07-18)

Source: Claude Code auto-memory `teja-force-pipeline-setup` (originally written
2026-06-05; `type: project`), migrated here so SDK agents can access it directly
without depending on the per-machine `~/.claude/projects/.../memory/` path.

- The TejaForce agent build pipeline lives in a git worktree at
  `/Users/tejaswi/Work/cogwheel-claude-dev` (sibling of the main repo) on
  branch `claude-dev` — `main` stays clean. The worktree has a scoped
  `core.hooksPath=.claude/hooks` and `ALLOWED_BRANCHES=["claude-dev"]` in
  `.claude/sdk/gates.py`; a pre-push guard blocks pushes to main/master.
- Conda env routing: the pipeline orchestrator AND agents run cogwheel code
  in **`cogwheel_310`** (Python 3.10, has both `cogwheel` and
  `claude-agent-sdk`) — NOT the main research env `cogwheel` (Python 3.9,
  incompatible with `claude-agent-sdk` >=3.10). `environment.yaml`'s
  declared `cogwheel-env` (python 3.12) does not exist on this machine —
  ignore it.
- SDK version pin: `claude-agent-sdk` must be **0.1.48** in `cogwheel_310`.
  The 0.2.x line (e.g. 0.2.119) times out on the SDK<->CLI handshake at the
  first live `query()` ("Control request timeout: initialize"). All local
  conda envs are x86_64 (Rosetta on an arm64 Mac); the Bun "CPU lacks AVX"
  warning is a red herring — the SDK version, not the arch, is the gate.
- Docs: cogwheel uses Sphinx (`docs/source/`, RST, autosummary in
  `api.rst`), built on Read the Docs.
- To run: from the worktree, `python .claude/sdk/cli.py build "task"` (or
  `/build`).
