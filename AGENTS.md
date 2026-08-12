# AGENTS.md — cogwheel

`cogwheel` (package `cogwheel-pe` on PyPI/conda, imported as `cogwheel`) is a
scientific Python library for Bayesian parameter estimation of gravitational-wave
sources. It implements a custom coordinate system, a "folding" algorithm for
multimodality, and the relative-binning likelihood (generalized to higher modes)
with marginalization over extrinsic parameters.

`CLAUDE.md` is a symlink to this file. All rules below are mandatory unless
flagged optional.

## Branch Safety
- Allowed branch for agent work: `claude-dev`. Run `git branch --show-current`
  before any project-changing command.
- Never switch branches automatically. Never commit on `main`/`master`. Never
  push to `main`/`master`.
- Never execute `scripts/sync_to_main.sh` — the user runs it manually.
- Agent-only paths excluded from that sync (`EXCLUDE_PATHS`): `.claude/`,
  `.codex/`, `.opencode/`, `.agents/`, `.serena/`, `.mcp.json`, `AGENTS.md`,
  `CLAUDE.md`. Keep agent-only content inside these.

## Environment
- Machine-specific settings (conda env, interpreter paths) go in untracked
  `.env` (copy `.env.example`). Set `SDK_CONDA_ENV` there; the build pipeline
  and git hooks consume it.
- Local-only overrides in untracked `CLAUDE.local.md` (never commit).
- Dependencies are in `pyproject.toml`; `environment.yaml` is minimal
  (python + pip only).

## Knowledge Anchoring
MUST read before any work that reads, modifies, or creates code or spec:
- `.claude/spec/SPEC.md` — mission, layered architecture, key abstractions
- `.claude/spec/TODO.md` — current and planned work
- `.claude/spec/DATA_CONTRACTS.yaml` — data-product contracts (units,
  conventions, producers/consumers)
- `.claude/spec/FINDINGS.md` — empirical gotchas (numerical tolerances,
  convention pitfalls, numba traps)

Package docs: `docs/source/` (Sphinx / Read the Docs); tutorials: `tutorials/`.

<!-- BEGIN AGENT INFRA SECTION — stripped by bootstrap_claude_workflow.sh for collaborators -->
- **Pipeline graph queries** (before modifying any data-producing/consuming
  module): `python scripts/pipeline_graph.py`:
  `resolve <artifact>` | `trace <artifact>` | `consumers_of <artifact>` |
  `inputs_for <Module/Class>`.
  Backed by `.claude/spec/DATA_CONTRACTS.yaml` + `.claude/spec/data_registry.yaml`.
<!-- END AGENT INFRA SECTION -->

## Engineering Values
Ordered by priority — this is a scientific numerical library:
1. **Correctness & numerical accuracy first.** Relative-binning and marginalized
   likelihoods MUST agree with exact/brute-force references within tolerance.
2. **Explicit over clever.** If it needs a comment to explain *what* it does,
   rewrite it.
3. **Respect conventions.** Units: frequencies Hz, times GPS seconds, masses
   solar masses, distances Mpc, angles radians. Waveform phase/spin conventions:
   IMRPhenomXP (**not** Pv2 — see LIGO-T1500602).
4. **DRY.** One authoritative representation per piece of knowledge.
5. **Well-tested.** Every public function and error path gets tests; numerically
   hot paths get tolerance-based accuracy tests.
6. **Engineered enough.** Prefer the simplest correct solution — neither fragile
   nor over-abstracted.
7. **numba-compatible.** Accelerated code (relative binning, coherent-score
   marginalization) must stay numba-compatible.

## Spec/TODO Workflow
Applies to **behavior changes** in `cogwheel/` (new functions, signature/logic/
control-flow changes, new data products). Housekeeping/auxiliary may skip with
tag `[housekeeping]`.

- **New work**: add fragment `.claude/spec/todo.d/<section>_<slug>.md`. Tag:
  `[→ spec]` | `[→ docs]` | `[housekeeping]`.
- **Completion**: delete the `todo.d/` fragment, add
  `.claude/spec/completed.d/<date>_<slug>.md` (frontmatter: `date`, `section`),
  update tagged doc(s).
- **SPEC.md edits**: include `.claude/spec/spec_changelog.d/<date>_<slug>.md`
  with `bump: patch|minor|major`. Never edit `spec_version`/`last_updated`
  directly — they are rendered.
- **DATA_CONTRACTS.yaml edits**: include `.claude/spec/contracts_changelog.d/
  <date>_<slug>.md` with `bump:`. Never edit `schema_version` directly.
- After writing any fragment: `python scripts/render_fragments.py`.

<!-- BEGIN AGENT INFRA SECTION — stripped by bootstrap_claude_workflow.sh for collaborators -->
## Shared Agent Pipeline
- Builds launch via `.claude/sdk/launch_build.sh <slug> <brief>` with
  `AGENT_PROVIDER=claude|codex|opencode`. No per-provider launch scripts.
- Claude Code: native settings, `.mcp.json`, hooks, commands, launch paths.
- Codex: `.codex/config.toml`, `.codex/hooks.json`, `.codex/agents/`.
  Serena Streamable HTTP on `CODEX_SERENA_PORT` (default `8324`).
- OpenCode: `.opencode/opencode.json`, `.opencode/agents/`, `.opencode/plugins/`.
  Serena Streamable HTTP on `OPENCODE_SERENA_PORT` (default `8325`).
- Model routing: Codex via `.codex/config.toml` (`CODEX_MODEL` /
  `CODEX_REASONING_EFFORT`); OpenCode via `.opencode/opencode.json`
  (`OPENCODE_MODEL` / `OPENCODE_VARIANT`). Agent frontmatter in
  `.opencode/agents/*.md` is auto-synced by `scripts/sync_opencode_agents.py`
  (called by `launch_build.sh`).
- Orchestration state, role contracts, specs, handoffs, and memories are
  shared under `.claude/` and `.serena/`; never fork per-provider copies.
- `.claude/sdk/runtime.py` is the provider boundary. `AGENT_PROVIDER` defaults
  to `claude`.

### Build launch protocol (all providers)

1. Write brief to `.claude/handoff/<slug>.md`.
2. Launch: Bash with `run_in_background: true`:
   `.claude/sdk/launch_build.sh <slug> .claude/handoff/<slug>.md`
   Never pass `--auto` unless the user explicitly says to.
3. When `<approval-dir>/plan_ready` appears, read `plan.json` and evaluate.
4. Approve: `touch <approval-dir>/plan_approved`.
   Reject: `echo "feedback" > <approval-dir>/plan_rejected`.
5. Monitor Phase 2 via log tail; watchdog kills stalled builds.

The driver reviews plans autonomously; escalate to the human user only when
genuinely unsure.

Monitor authoring: emit on a NEW OCCURRENCE, never on "the newest matching
line differs from the last one I emitted". Track the COUNT of matching lines
and print the ones past the previous count. Repeated identical events are real
events, and the `[file-based]` log lines carry NO timestamp to tell them
apart — `[file-based] Plan written to <dir>/plan.json` is byte-identical on a
replan. MEASURED 2026-08-12: a last-line-comparison monitor stayed silent
through the second plan_ready of a rejected-then-resubmitted plan, and the
build sat waiting on driver approval for 8 minutes with the watchdog staleness
clock running; a tighter threshold would have killed a healthy build because
the watcher could not see a repeat. Two monitor blind spots are now on record
— this one, and self-matching the monitor's own command echoed into the log
(filter it with `grep -av "Monitor(persistent"`).
When verifying a monitor fix, reproduce the miss on the REAL line format
first: a probe built from synthetic timestamped lines does not recreate a
byte-identical-repeat bug and will "pass" against a fix that does nothing.

### Quiet build monitoring (Codex / OpenCode)
- Never create a persistent goal to monitor a build. Never model-poll a log.
  The driver acts only after an escalation or terminal callback.
- Codex: when `CODEX_THREAD_ID` is inherited, events resume via
  `.codex/resume_driver.sh`.
- OpenCode: when `OPENCODE_SESSION_ID` is inherited, events resume via
  `.opencode/resume_driver.sh`.

### SDK Scripts (driver toolbox)
- `launch_build.sh <slug> <prompt_file>` — ONLY sanctioned build launch.
  Attaches watchdog; prints log path + post-build sequence.
- `watchdog.sh <log> [stale_s] [pid]` — auto-attached by launcher; kills
  orchestrator subtree on stall. Do NOT invoke by hand.
- `stale_alarm.sh <log> [stale_s]` — alert on silence, no kill.
- `verify_watchdog.sh` — ~12 s probe after touching watchdog/launcher code.
- `run_full_suite.sh [log]` — fast gate: collect-count, xdist (`-n 8
  --dist loadfile`) minus timing guards, then timing guards serial.
  `EXPECT_COLLECTED=<n>` fails on collection mismatch.
- `post_build_sweeps.sh` — slow tiers, one process per file. NEVER in a build.
- `timing_pass.sh` — serial timing guards only.

### Build Briefs discipline
- Briefs contain: mission, scope fences, measured facts agents cannot obtain
  (~15 lines), build-level acceptance, constraints. No WP decompositions or
  history.
- Quantitative "measured facts" MUST carry the SHA measured at.
- ONE level of work tracking: `.claude/spec/todo.d/` fragments, rendered to
  `TODO.md`. Depends_on ordering on fragments; renderer topologically sorts.
- Prefer several small sequential builds over one wide one.
- In-build tests must be FAST (small/synthetic configs). Bulk sweeps are
  POST-BUILD driver steps.
<!-- END AGENT INFRA SECTION -->

## Testing
- Tests live in `cogwheel/tests/` (stdlib `unittest`), **not** a top-level
  `tests/`.
- Run all fast tests: `python -m pytest cogwheel/tests/ -v`
- Run one file: `python -m pytest cogwheel/tests/test_prior.py -v`
- Run one test: `python -m pytest cogwheel/tests/test_prior.py::PriorTestCase::test_m1m2 -v`
- Full suite gate: `python -m pytest cogwheel/tests/ -q -n 8 --dist loadfile
  -k "not Timing and not timing"`, then deselected timing guards in one serial
  pass. First: `--collect-only -q` to verify collected count.
- New tests go in `cogwheel/tests/`. Cover numerical-accuracy paths with
  tolerance-based assertions.
- **Setup gotcha**: fresh worktrees lack the untracked
  `cogwheel/waveform_models/IMRPhenomXODE` symlink; without it
  `test_waveform`/`test_posterior`/`test_gw_prior` silently collect-error.
- Conda env: set `SDK_CONDA_ENV` in untracked `.env`.
- numba cache: xdist workers share one `NUMBA_CACHE_DIR`; independent
  concurrent pytest processes each need their own cache dir.
- `conftest.py` applies a 900 s per-test timeout on fast-tier runs (lifts when
  any slow-tier env var is set). Uses pytest-timeout — silent no-op if absent.

### Assert VALUES, not code paths
- Assert **what the answer is**, against an oracle and a tolerance — not which
  branch produced it. Value claims survive refactors; path claims break.
- Each routing decision gets ONE canonical pin in the file that owns the
  predicate. Don't re-assert it across consumer suites.
- Before adding a path/exception-type assertion: ask *"if this code were wrong
  but still took this path, would this test fail?"* If no, write a value
  assertion instead.

### Test tiers
- Default: fast tier only. Slow tests skip with a loud reason.
- Slow tiers opt-in by env var: `COGWHEEL_BRUTE_ACCURACY=1` (brute-force
  accuracy), `COGWHEEL_STRICT_TIMING=1` (timing guards).
<!-- BEGIN AGENT INFRA SECTION — stripped by bootstrap_claude_workflow.sh for collaborators -->
- Slow tests NEVER run inside a build (SDK pins gate vars empty).
- Agents verify ONLY tests they changed; driver runs full tally post-build.
- Detached-run health: watch worker CPU + log growth. xdist master idles at
  0% legitimately. Identify workers via `ps aux | grep "cogwheel"` — xdist
  worker argv is `python -u -c import sys;exec(...)`.
- Bracket idiom: `pgrep -f "pytest [c]ogwheel"` or the pattern matches itself.
- Never edit a script in place while a shell is executing it; patch via
  sidecar + atomic replace (`os.replace` / `mv`).
<!-- END AGENT INFRA SECTION -->

## Generated Files
- `CHANGELOG.md` is generated by `scripts/render_fragments.py` — do not edit
  directly. New entry: fragment `changelog.d/<date>_<slug>.md` with frontmatter
  `date: YYYY-MM-DD`.
- `scripts/sync_derived_docs.py` auto-syncs derived outputs; the `.claude/hooks/
  pre-commit` hook runs it and auto-stages changes. Do not hand-edit generated
  outputs.
- After writing any fragment: `python scripts/render_fragments.py`.

## Pre-commit Hooks
Custom hooks in `.claude/hooks/pre-commit` (installed via
`.claude/hooks/install_hooks.sh`) enforce:
- Branch safety (block `main`/`master`)
- Staged Python must parse
- Gated-test drift detection
- Retired concept enforcement
- Spec/doc changelog discipline (fragment requirements)
- Fragment rendering + auto-stage of canonicals
- Derived docs auto-sync
In SDK builds, documentation-hygiene failures are deferred (recorded to
`.claude/doc_debt.json`) rather than blocking the commit.

<!-- BEGIN SERENA SECTION — stripped by bootstrap_claude_workflow.sh for non-Serena users -->
## Serena Tools
Serena MCP is available. Use it for symbolic navigation, search, and edits
to minimize context.

- **Before any code or spec work**: call `mcp__serena__initial_instructions`.
- Read: `read_file` / `find_symbol(include_body=True)`. Never `read_file` with
  guessed line offsets.
- File orientation: `get_symbols_overview` before whole-file read.
- Search: `search_for_pattern` (regex), `find_symbol` (by name),
  `find_file` / `list_dir` (by name/glob).
- Edit: `replace_symbol_body` (whole symbol), `replace_content` (find-replace),
  `replace_lines` / `delete_lines` / `insert_at_line` (line-range).
  Insert before/after definitions: `insert_before_symbol` /
  `insert_after_symbol`. New file: `create_text_file`.
- Rename: `rename_symbol` across codebase.
- **Find all callers before moving/deleting/renaming**: `find_referencing_symbols`
  **and** a `search_for_pattern` grep for `\.method_name\(`. LSP misses
  cross-file refs silently — empty results REQUIRE the grep cross-check.
- After any non-Serena edit, restart the language server.
- Shell: `execute_shell_command`.

## Insertion Safety
- Symbol `end_line` from the LSP is unreliable (truncates at early returns,
  `if`/`else` final lines, dict literals in docstrings).
- Before line-based insertion: read ~5 lines on each side of target.
- After mid-file insertion: search for displaced/duplicate fragments and remove
  them.
<!-- END SERENA SECTION -->
