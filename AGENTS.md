This is the canonical shared instruction file for Claude Code, Codex, and OpenCode.
`CLAUDE.md` is a compatibility symlink to this file. All rules below are
mandatory unless flagged optional.

## Branch Safety
- Allowed working branch for agent work: `claude-dev`. Before editing or running project-changing commands, run `git branch --show-current`.
- Never switch branches automatically. Never commit on `main`/`master`.

## Cross-Branch / Master Safety
- Never push to `main`/`master`. No local override permitted.
- Never execute `scripts/sync_to_main.sh` — the user runs it manually to merge code into `main`.
- Agent-only paths excluded from that sync (`EXCLUDE_PATHS` in `scripts/sync_to_main.sh`): `.claude/`, `.codex/`, `.opencode/`, `.agents/`, `.serena/`, `.mcp.json`, `AGENTS.md`, and `CLAUDE.md`. Keep agent-only content inside these.

## Local Override Convention
- Tracked `AGENTS.md` stays portable and repo-relative; `CLAUDE.md` remains
  its tracked symlink for Claude Code compatibility.
- Machine-specific instructions (conda env name, absolute interpreter/data paths, cluster details) go in untracked `CLAUDE.local.md` (never commit). Copy `CLAUDE.local.md.example` to start.

## Knowledge Anchoring
MUST read before any work that reads, modifies, or creates code or spec:
- `.claude/spec/SPEC.md` — mission, layered architecture, key abstractions, constraints
- `.claude/spec/TODO.md` — current and planned work
- `.claude/spec/DATA_CONTRACTS.yaml` — machine-readable data-product contracts (units, conventions, producers/consumers)
- `.claude/spec/FINDINGS.md` — empirical gotchas (numerical tolerances, convention pitfalls, numba traps)

Package docs live under `docs/source/` (Sphinx / Read the Docs); runnable tutorials in `tutorials/`.

<!-- BEGIN AGENT INFRA SECTION — stripped by bootstrap_claude_workflow.sh for collaborators -->
- **Pipeline graph queries** (before modifying any data-producing/consuming module), via `python scripts/pipeline_graph.py`:
  - `resolve <artifact>` — what produces an artifact
  - `trace <artifact>` — full producer/consumer chain + disk paths
  - `consumers_of <artifact>` — all code that reads an artifact
  - `inputs_for <Module/Class>` — what artifacts a module consumes
  Backed by `.claude/spec/DATA_CONTRACTS.yaml` + `.claude/spec/data_registry.yaml`.
<!-- END AGENT INFRA SECTION -->

## Engineering Values
Ordered by priority when they conflict — this is a scientific numerical library, so accuracy dominates:
1. **Correctness & numerical accuracy first.** Relative-binning and marginalized likelihoods MUST agree with exact / brute-force references within tolerance. A fast result that is wrong is worthless.
2. **Explicit over clever.** If it needs a comment to explain *what* it does, rewrite it.
3. **Respect conventions.** Units (frequencies Hz, times GPS seconds, masses solar masses, distances Mpc, angles radians) and waveform phase/spin conventions (IMRPhenomXP, **not** Pv2 — see LIGO-T1500602) must hold across waveform and coordinate code.
4. **DRY.** One authoritative representation per piece of knowledge.
5. **Well-tested.** Every public function and error path gets tests; numerically hot paths get tolerance-based accuracy tests.
6. **Engineered enough.** Neither fragile/hacky nor over-abstracted — prefer the simplest correct solution.
7. **numba-compatible.** Accelerated code (relative binning, coherent-score marginalization) must stay numba-compatible.

## Spec/TODO Workflow
Applies to **behavior changes** in `cogwheel/` (new functions, signature/logic/control-flow changes, new data products). Auxiliary work (repo management, housekeeping, docs-only) may skip the workflow; tag those TODOs `[housekeeping]`.

- **New work**: add a fragment `.claude/spec/todo.d/<section>_<slug>.md`. Tag one of:
  - `[→ spec]` — update `.claude/spec/SPEC.md`
  - `[→ docs]` — update the relevant page under `docs/source/`
  - `[housekeeping]` — no doc update needed
- **Completion**: delete the `todo.d/` fragment, add `.claude/spec/completed.d/<date>_<slug>.md` (frontmatter: `date`, `section`), and update the tagged doc(s).
- **SPEC.md edits**: include a `.claude/spec/spec_changelog.d/<date>_<slug>.md` fragment with `bump: patch|minor|major` (semver). Never edit `spec_version` / `last_updated` directly — they are rendered.
- **DATA_CONTRACTS.yaml edits**: include a `.claude/spec/contracts_changelog.d/<date>_<slug>.md` fragment with `bump:`. Never edit `schema_version` directly.
- After writing any fragment: `python scripts/render_fragments.py`.

<!-- BEGIN AGENT INFRA SECTION — stripped by bootstrap_claude_workflow.sh for collaborators -->
## Shared Agent Pipeline
- Claude Code keeps its existing native settings, `.mcp.json`, hooks, commands,
  and launch paths. `.claude/build` defaults to the Claude Agent SDK.
- Codex uses `.codex/config.toml`, `.codex/hooks.json`, `.codex/agents/`, and
  `.agents/skills/`. Launch the same state machine with `.codex/build`. A
  Codex build starts one shared Serena Streamable HTTP server on
  `CODEX_SERENA_PORT` (default `8324`) and reuses its warm index for every
  build role.
- OpenCode uses `.opencode/opencode.json`, `.opencode/agents/`, and
  `.opencode/plugins/`. Launch the same state machine with `.opencode/build`. An
  OpenCode build starts one shared Serena Streamable HTTP server on
  `OPENCODE_SERENA_PORT` (default `8325`) and reuses its warm index for every
  build role.
- Codex routes the Architect and planning Professor to `gpt-5.6-sol` at high
  reasoning. Coder, Test Developer, Inspector, and ProfReview use
  `gpt-5.6-terra` at high reasoning; administrative support roles use Terra at
  medium reasoning. `CODEX_MODEL` /
  `CODEX_REASONING_EFFORT` override all roles; suffixed variables such as
  `CODEX_MODEL_TEST_DEV` override one role. Claude's native role map is
  unchanged.
- OpenCode routes the Architect, Coder, Inspector, Professor, and ProfReview to
  `claude-v4.6-opus` at high variant. Foreman-Lite, Test Developer, Librarian,
  Tidier, Dreamer, and Simplifier use `claude-v4.6-sonnet`. `OPENCODE_MODEL` /
  `OPENCODE_VARIANT` override all roles; suffixed variables such as
  `OPENCODE_MODEL_TEST_DEV` override one role.
- The orchestration state, role contracts, specs, handoffs, and memories remain
  shared under `.claude/` and `.serena/`; never fork provider-specific copies.
- `.claude/sdk/runtime.py` is the provider boundary. `AGENT_PROVIDER` defaults
  to `claude`; normal Claude resumes therefore retain their prior behavior.
  Set `AGENT_PROVIDER=codex` or `AGENT_PROVIDER=opencode` for those backends.

### Codex quiet build monitoring

- `.codex/build` is the sanctioned Codex build launcher. It attaches the same
  shared SDK watchdog as Claude, verifies that attachment, and retains the
  normal watchdog terminal/stale-kill behavior.
- When `CODEX_THREAD_ID` is inherited, terminal build events and file-based
  escalation events resume that exact thread through `.codex/resume_driver.sh`.
  Events are occurrence-unique and serialized per thread; a queued event is
  never discarded because another callback is active.
- Never create a persistent Codex goal to monitor a build. Never model-poll a
  log or wait on a timer. The driver acts only after an escalation or terminal
  callback. Set `CODEX_EVENT_RESUME=0` only to intentionally opt out.

### OpenCode quiet build monitoring

- `.opencode/build` is the sanctioned OpenCode build launcher. It attaches the
  same shared SDK watchdog as Claude/Codex, verifies that attachment, and
  retains the normal watchdog terminal/stale-kill behavior.
- When `OPENCODE_SESSION_ID` is inherited, terminal build events and file-based
  escalation events resume that exact session through
  `.opencode/resume_driver.sh`. Events are occurrence-unique and serialized per
  session; a queued event is never discarded because another callback is active.
- Never create a persistent OpenCode goal to monitor a build. Never model-poll a
  log or wait on a timer. The driver acts only after an escalation or terminal
  callback. Set `OPENCODE_EVENT_RESUME=0` only to intentionally opt out.

## SDK Build Briefs (driver discipline)
Transcript depth is a reliability constraint: the auto-mode permission
classifier fails closed more often as agent transcripts deepen (measured
2026-07: 0/106 bare denials in the first two tool calls of a session; median
at call 14; see claude-code issue #74351 — no upstream fix). Keep builds
shallow:
- A brief contains: mission, in/out scope fences, measured facts the agents
  cannot obtain themselves (inline, ~15 lines), build-level acceptance, and
  constraints. It does NOT contain WP decompositions (the Architect owns
  decomposition), history/narrative (reference files instead), or pointers
  to `.claude/handoff/**/META_PLAN.md` (driver journal, never agent
  context).
- Live documents (META_PLAN, plans, configs, briefs) state CURRENT truth
  only: rewrite superseded content in place — git is the archive. Never
  append corrections atop stale entries.
- NO engine-run launch (campaign, pilot, sweep, probe) without a cost
  estimate computed first — unit count x measured per-unit cost, quoted
  in the launch message. Applies to every config change, including
  expert-authorized ones and the driver's own launches.
- Prefer several small sequential builds over one wide one. If an honest
  decomposition needs more than ~3 WPs, split into sequential builds; reject
  over-wide plans at the plan gate.
- Style calibration: single-focus briefs of 5-9 KB (the gw pipeline's
  37-build precedent — median ~9 agents/build, no classifier denials).
- Two-tier verification (gw-proven): in-build tests must be FAST —
  small/synthetic configurations, analytic or few-eval oracles — so every
  claim has a falsifiable gate agents can actually run. Bulk-data sweeps
  and hour-scale regressions are POST-BUILD driver steps, named in the
  brief's acceptance ("full suite green, driver-verified post-build"),
  never in-build test specs. A test spec that takes an hour to run is a
  build-killer (deep transcripts) and an unverifiable gate (inspectors
  cannot run it).

<!-- END AGENT INFRA SECTION -->

## Testing
- Tests live in `cogwheel/tests/` (stdlib `unittest`), **not** a top-level `tests/`.
- Run: `python -m pytest cogwheel/tests/ -v` (or `python -m unittest discover -s cogwheel/tests`).
- New tests go beside the suite in `cogwheel/tests/`. Cover numerical-accuracy paths with tolerance-based assertions.

### Assert VALUES, not code paths
- A test should assert **what the answer is**, against an oracle and a tolerance — not **which branch produced it**. Value claims survive refactors and catch bugs; structural claims break on every refactor and catch nothing.
- Each routing decision gets **ONE canonical pin**, in the file that owns the predicate. Do not re-assert the same decision in every consumer suite. `test_lensing_operator.py::test_thresholds_have_one_home` is the model.
- Before adding a test that asserts a served path, an exception type, or an internal call, ask: *if this code were wrong but still took this path, would this test fail?* If no, write the value assertion instead.
- MEASURED COST (2026-07-29): the `select_branch` decision was pinned in 16 test methods across 6 files and `SchwingerCertificationError` identity in 32 methods across 10. Changing two branch conditions therefore re-pointed eight files over three revision rounds — about two-thirds of that build's wall clock — and none of those tests had caught the two real defects (F028, F029), because every one of them asserted the path rather than the number.
- Conda env is machine-specific: set `SDK_CONDA_ENV` in untracked `.env` (copy `.env.example`); default `cogwheel_310`.

### Test tiers
- Default runs execute the fast tier only; slow tests skip with a loud reason.
- Slow tiers are opt-in by env var: brute-force accuracy `COGWHEEL_BRUTE_ACCURACY=1`; strict timing `COGWHEEL_STRICT_TIMING=1`.

<!-- BEGIN AGENT INFRA SECTION — stripped by bootstrap_claude_workflow.sh for collaborators -->
### Test tiers — driver/agent discipline
- Slow tests NEVER run inside a build. No exceptions. (Enforced: the SDK pins both gate vars empty in every agent env.)
- Slow sweeps are the driver's post-build parallel job: `.claude/sdk/post_build_sweeps.sh` (one process per file, per-process numba cache).
- Agents verify ONLY the tests they changed. The driver runs the full tally once per build.
- Every long run emits a countable progress stream: pytest `-v` teed to a log + a Monitor reporting percent/rate/projected finish. Zero progress across two beats = investigate with py-spy, never wait. A run without a progress monitor is unattended, not monitored.
- MECHANICAL PAIRING: the same response that launches a background run arms its progress monitor. Launch and monitor are one action, never two. Long-running scripts additionally SELF-EMIT progress beats on stdout so observation needs no instrumentation.
- Monitors emit on CHANGE only: a beat fires when the progress count moves, ONCE on entering a stall, and at terminal — never on an unchanged interval. Poll internally as often as needed; each EMITTED line costs a driver invocation. Scale the poll interval to the run (minutes-scale run: 1-2 min; hour-scale: 10-15 min).

### SDK scripts — the driver's whole toolbox
Nothing here needs re-deriving per run. If you are about to hand-roll a launch, a
gate command, or a watch loop, one of these already does it.
- `launch_build.sh <slug> <prompt_file> [stale_s]` — the ONLY sanctioned build launch. Starts the orchestrator, attaches the watchdog, verifies it attached, prints the log path and the post-build sequence.
- `watchdog.sh <log> [stale_s] [pid]` — auto-attached by the launcher; kills the orchestrator subtree when the log stops advancing. Logs to `<log>.watchdog.log`. Do NOT invoke by hand for a launcher-started build.
- `verify_watchdog.sh` — ~12 s probe proving the watchdog actually kills a stalled build and that its fallback pattern still matches the launcher's entrypoint. Run it after touching either script (F055: it failed open for three days while the launcher reported it armed).
- `stale_alarm.sh <log> [stale_s]` — exits when the log goes quiet, converting silence into a task notification. Use when you want an alert but NOT a kill.
- `run_full_suite.sh [log]` — the post-build fast gate: collect-count check, then `-n 8 --dist loadfile` with timing guards deselected, then those guards in one serial pass. Self-emits `[beat] n/N (x%)`. Slow tiers pinned OFF. `EXPECT_COLLECTED=<n>` fails the gate on a collection mismatch.
- `post_build_sweeps.sh` — the slow tiers, one process per file. NEVER in a build.
- `timing_pass.sh` — the serial timing guards on their own.
- A stalled watcher and a finished one look identical. Prefer a watcher that EXITS on its terminal condition over one that reports progress forever.
<!-- END AGENT INFRA SECTION -->

### Full-suite gate
- `python -m pytest cogwheel/tests/ -q -n 8 --dist loadfile -k "not Timing and not timing"`, then the deselected timing guards in one serial pass. Never serial for the whole gate.
- First: `--collect-only -q`; the collected count must match expectation.
- numba cache: xdist workers share one `NUMBA_CACHE_DIR`; independent concurrent pytest processes get one cache dir EACH.

### Detached-run health
- Health = worker CPU + log growth. An xdist master legitimately idles at 0%.
- xdist worker argv is `python -u -c import sys;exec(...)` — it does not contain "pytest".
- `pgrep -f`/`pkill -f`: bracket idiom (`pgrep -f "pytest [c]ogwheel"`) or the check matches itself.
- Kill a run only on ~10x contention-adjusted overshoot or zero log growth.
- Never write in place to a script a live shell may be executing (bash reads incrementally; a mid-run edit corrupts the running instance). Patch via sidecar + atomic replace (os.replace / mv).
- Fresh worktrees lack the untracked `cogwheel/waveform_models/IMRPhenomXODE` symlink; recreate it or `test_waveform`/`test_posterior`/`test_gw_prior` silently collect-error.

## CHANGELOG.md Invariant
- `CHANGELOG.md` is generated by `scripts/render_fragments.py` — do not edit it directly.
- New entry: a fragment `changelog.d/<date>_<slug>.md` with frontmatter `date: YYYY-MM-DD`; body is a `### heading` plus prose.
- Run `python scripts/render_fragments.py` after writing fragments.

## Derived Docs
- Some docs/outputs are generated; `scripts/sync_derived_docs.py` keeps them in sync (the pre-commit hook runs `--check` and then auto-fixes/auto-stages). Do not hand-edit generated outputs.

<!-- BEGIN SERENA SECTION — stripped by bootstrap_claude_workflow.sh for non-Serena users -->
## Serena Tools
Serena MCP is available (project `cogwheel`). Interactive Claude uses the
`claude-code` context from `.mcp.json`; interactive Codex uses the `codex`
context from `.codex/config.toml`; interactive OpenCode uses the `claude-code`
context from `.opencode/opencode.json` (OpenCode's tool surface matches Claude
Code's, so the same Serena context applies). Builds use a separate, build-scoped
Serena server: Claude uses SSE on `SDK_SERENA_PORT` (default `8322`, locally
`8323`); Codex uses Streamable HTTP on `CODEX_SERENA_PORT` (default `8324`);
OpenCode uses Streamable HTTP on `OPENCODE_SERENA_PORT` (default `8325`). Each
server is shared by every role in that build. Prefer Serena for symbolic
navigation, search, and edits because it minimizes context and preserves
reference awareness. Use a client-native exact patch when Serena cannot
represent the change. Intent -> tool:
- If Codex has not yet surfaced `mcp__serena__*`, use its `tool_search` to
  discover Serena tools, then call `mcp__serena__initial_instructions` before
  code or spec work. Native Codex project tools are blocked per interactive
  thread or build role process until their corresponding Serena initialization
  completes.
- OpenCode: call `mcp__serena__initial_instructions` before code or spec work.
  The `.opencode/plugins/cogwheel-hooks.ts` plugin tracks readiness and routes
  python commands through conda.
- Read file / symbol body: `read_file` / `find_symbol(include_body=True)`. Never `read_file` with guessed offsets.
- File orientation before any whole-file read: `get_symbols_overview`.
- Text/regex search: `search_for_pattern`. Symbol-name search: `find_symbol`. File discovery: `find_file` / `list_dir`.
- Whole-symbol edit: `replace_symbol_body`. Targeted find/replace: `replace_content`. Line-range: `replace_lines` / `delete_lines`. Insert: `insert_at_line` (or `insert_before_symbol` / `insert_after_symbol`). New file: `create_text_file`. Rename across codebase: `rename_symbol`.
- **Find all callers before moving/deleting/renaming a symbol**: `find_referencing_symbols` **and** a `search_for_pattern` grep for `\.method_name\(`. The LSP misses cross-file refs silently — empty results REQUIRE the grep cross-check. Mandatory.
- Shell: `execute_shell_command`. After any non-Serena code edit, restart the language server.

## Insertion Safety
- Symbol `end_line` from the LSP is unreliable (truncates at early returns, `if/else` final lines, or docstrings with dict literals).
- Before any line-based insertion: read ~5 lines on each side of the target to confirm it sits between top-level definitions, not inside a body.
- After any mid-file insertion: search the file for displaced/duplicate fragments and remove them.
<!-- END SERENA SECTION -->
