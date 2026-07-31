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
  `.env.example`); `.env` is always authoritative — no hardcoded fallback.
  `.claude/sdk/launch_build.sh` and `.claude/build` source it; the launcher
  then resolves the env python absolutely and fails loudly if missing.
- The TejaForce agent build pipeline lives in a git worktree named
  `cogwheel-claude-dev` (sibling of the main repo) on branch `claude-dev` —
  `main` stays clean. The worktree has a scoped
  `core.hooksPath=.claude/hooks` and `ALLOWED_BRANCHES=["claude-dev"]` in
  `.claude/sdk/gates.py`; a pre-push guard blocks pushes to main/master.
- Conda env routing: the pipeline orchestrator AND agents run cogwheel code
  in the environment selected by `SDK_CONDA_ENV`. There is NO hardcoded
  fallback — `.env` must be set (copy `.env.example`). This IAS worktree uses
  **`cogwheel-newlal`** (Python 3.10, has both `cogwheel` and
  `claude-agent-sdk`) — NOT the main research env `cogwheel` (Python 3.9,
  incompatible with `claude-agent-sdk` >=3.10). Scripts fail loud if `.env`
  is missing or `SDK_CONDA_ENV` is unset.
- Installed Claude runtime: `claude-agent-sdk` is **0.1.53** in the active
  `cogwheel-newlal` environment (verified 2026-07-24, but still not declared
  as a project dependency). The 0.2.x line (e.g. 0.2.119) times out on the
  SDK<->CLI handshake at the first live `query()` ("Control request timeout:
  initialize"). All local conda envs are x86_64 (Rosetta on an arm64 Mac);
  the Bun "CPU lacks AVX" warning is a red herring — the SDK version, not the
  arch, is the gate.
- `conda run -n <env>` IS NOT RELIABLE on the IAS server and must not be used
  to run project code. `CONDA_ENVS_PATH=/scratch/lustre/tejaswi/conda_envs` is
  set in the shell profile but that directory is EMPTY, so `conda run -n
  cogwheel-newlal` finds no env and silently falls back to whatever `python`
  PATH gives — which is the uv-provisioned interpreter under
  `/scratch/lustre/tejaswi/.cache/uv/...` with no numpy/scipy/numba. The
  symptom is a baffling "numpy MISSING" for an env that plainly has numpy.
  Always resolve the interpreter absolutely:
  `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`.
  `.claude/sdk/launch_build.sh` already does this deliberately (see its
  comment at the `PYBIN=` line), so BUILDS are unaffected — only ad-hoc shell
  calls are. Serena's `execute_shell_command` inherits that same PATH, so a
  bare `python` there is the uv interpreter, not the project env.
- Read `.env` FIRST; there is no fallback default. Scripts fail immediately
  if `SDK_CONDA_ENV` cannot be resolved from `.env`.
- Scientific extras present in `cogwheel-newlal` (verified 2026-07-29):
  mpmath 1.3.0, sympy 1.14.0, numba 0.58.1, numpy 1.26.2.
- Docs: cogwheel uses Sphinx (`docs/source/`, RST, autosummary in `api.rst`),
  built on Read the Docs.
- Provider layout (2026-07-24): `AGENTS.md` is canonical and `CLAUDE.md`
  symlinks to it. Claude keeps `.mcp.json`, `.claude/settings.json`, and
  `.claude/build`; Codex uses `.codex/config.toml`, `.codex/hooks.json`, and
  `.codex/build`. Both backends share `.claude/sdk`, crew contracts, specs,
  handoffs, and these Serena memories.
- Serena build lifecycle (verified 2026-07-24): interactive sessions own
  independent stdio Serena processes. Each multi-agent build owns one warm
  Serena process shared by all roles. Claude uses legacy SSE on
  `SDK_SERENA_PORT` (this worktree: 8323). Current Codex treats every MCP
  `url=` as Streamable HTTP (official config reference) and therefore MUST
  start Serena with `--transport streamable-http` and connect to `/mcp` on
  `CODEX_SERENA_PORT` (default 8324); pointing Codex at Serena's legacy `/sse`
  endpoint deterministically fails the initialize POST with HTTP 405. Distinct
  ports prevent simultaneous Claude/Codex builds from bind or watchdog
  collisions.
- Codex event-stream invariant (verified 2026-07-24): `codex exec --json`
  emits newline-delimited JSON, and one tool-completion event may embed an
  entire large file. Python asyncio's 64 KiB default StreamReader limit is too
  small and raises `ValueError: Separator is found, but chunk is longer than
  limit`. `.claude/sdk/runtime_codex.py` passes an 8 MiB reader limit; the
  portable escape hatch is `CODEX_JSON_STREAM_LIMIT` in `.env`.
- Codex build role routing: authority roles use `gpt-5.6-sol/high`; bounded
  support roles use `gpt-5.6-terra/medium`. Precedence is an explicit one-off
  model override, then role-specific env (`CODEX_MODEL_TEST_DEV` etc.), then
  global env, then the role default. Build logs must report the provider-
  effective model, not the legacy Claude role label.
- If an outer Codex filesystem sandbox makes the machine's uv cache
  (`/scratch/lustre/tejaswi/.cache/uv` here) read-only, Serena's `uvx` startup
  fails before binding. Relaunching the build with the approved scoped
  `.codex/build` escalation is the correct boundary; `/scratch/lustre` is a
  package cache, never the project root.
- Agent failures must retain the final bounded stderr/result detail in the
  build log. A length-only message such as "partial output: 1167 chars" is not
  sufficient for provider-boundary diagnosis.
- POST-OUTAGE CONCURRENCY DEATHS (diagnosed 2026-07-29): after an API outage,
  builds die in Phase 1 Planning with "Fatal error in message reader: Command
  failed with exit code 1" ~15-40s in, at "Architect planning (with Professor
  + Simplifier subagents)". A single `claude -p` call succeeds throughout, so
  it is CONCURRENCY limiting, not a full outage: the Architect's on-demand
  nested Task-tool subagents (professor/simplifier, `build_phase1_subagents`)
  spawn fresh CLI subprocesses, and a transient rate-limit failure on one of
  those kills the Architect's message stream and the whole build.
  KEY GAP: the orchestrator's infrastructural-death retry
  (`_looks_infrastructural` matches "exit code 1"/"command failed";
  `SDK_AGENT_RETRY_WAIT_SECONDS` default 300) wraps TOP-LEVEL role agents only,
  NOT the nested planning subagents — so these deaths are never retried and
  abort the build. Remedy in the moment is TIME (retry the whole build on a
  ~15 min cadence until the limit clears); do NOT disable the planning
  subagents to dodge it (that ships plans with no Professor review, which this
  session repeatedly relied on to catch driver-brief errors). Durable fix for
  later: extend the infra-death retry to cover nested Phase 1 subagents.
- RETRY-LOOP DETECTION TRAP (2026-07-29): a driver retry loop that greps the
  build LOG for survival markers ("Plan written", "plan_ready", "Coder
  checkpoint") FALSE-POSITIVES, because `launch_build.sh` echoes a suggested
  Monitor command into the log header containing those exact words. Key
  survival on the ARTIFACT instead — `/tmp/<slug>_approval/plan.json` existing
  — and death on real lines absent from that header echo ("Fatal error in
  message reader", a line starting "Build failed:").
- To run: `AGENT_PROVIDER=<provider> .claude/sdk/launch_build.sh <slug> <brief>`
  for all providers. The unified launcher handles conda, watchdog, approval-dir,
  and disown. There are no separate per-provider build scripts.
