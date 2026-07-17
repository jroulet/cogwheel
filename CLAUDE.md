All rules below are mandatory unless flagged optional.

## Branch Safety
- Allowed working branch for agent work: `claude-dev`. Before editing or running project-changing commands, run `git branch --show-current`.
- Never switch branches automatically. Never commit on `main`/`master`.

## Cross-Branch / Master Safety
- Never push to `main`/`master`. No local override permitted.
- Never execute `scripts/sync_to_main.sh` — the user runs it manually to merge code into `main`.
- Agent-only paths excluded from that sync (`EXCLUDE_PATHS` in `scripts/sync_to_main.sh`): `.claude/`, `.serena/`, `.mcp.json`. Keep agent-only content inside these.

## Local Override Convention
- Tracked `CLAUDE.md` stays portable and repo-relative.
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

## Testing
- Tests live in `cogwheel/tests/` (stdlib `unittest`), **not** a top-level `tests/`.
- Run: `python -m pytest cogwheel/tests/ -v` (or `python -m unittest discover -s cogwheel/tests`).
- New tests go beside the suite in `cogwheel/tests/`. Cover numerical-accuracy paths with tolerance-based assertions.
- (The conda env to run under is machine-specific — set `SDK_CONDA_ENV` in an untracked `.env` at the repo root; copy `.env.example` to start. Defaults to `cogwheel_310` when unset.)

## CHANGELOG.md Invariant
- `CHANGELOG.md` is generated by `scripts/render_fragments.py` — do not edit it directly.
- New entry: a fragment `changelog.d/<date>_<slug>.md` with frontmatter `date: YYYY-MM-DD`; body is a `### heading` plus prose.
- Run `python scripts/render_fragments.py` after writing fragments.

## Derived Docs
- Some docs/outputs are generated; `scripts/sync_derived_docs.py` keeps them in sync (the pre-commit hook runs `--check` and then auto-fixes/auto-stages). Do not hand-edit generated outputs.

<!-- BEGIN SERENA SECTION — stripped by bootstrap_claude_workflow.sh for non-Serena users -->
## Serena Tools
Serena MCP is available (project `cogwheel`). Use Serena for all code ops. Intent -> tool:
- Read file / symbol body: `read_file` / `find_symbol(include_body=True)`. Never `read_file` with guessed offsets.
- File orientation before any whole-file read: `get_symbols_overview`.
- Text/regex search: `search_for_pattern`. Symbol-name search: `find_symbol`. File discovery: `find_file` / `list_dir`.
- Whole-symbol edit: `replace_symbol_body`. Targeted find/replace: `replace_content`. Line-range: `replace_lines` / `delete_lines`. Insert: `insert_at_line` (or `insert_before_symbol` / `insert_after_symbol`). New file: `create_text_file`. Rename across codebase: `rename_symbol`.
- **Find all callers before moving/deleting/renaming a symbol**: `find_referencing_symbols` **and** a `search_for_pattern` grep for `\.method_name\(`. The LSP misses cross-file refs silently — empty results REQUIRE the grep cross-check. Mandatory.
- Shell: `execute_shell_command`. After any non-Serena code edit (`Edit`/`Write`), restart the language server.

## Insertion Safety
- Symbol `end_line` from the LSP is unreliable (truncates at early returns, `if/else` final lines, or docstrings with dict literals).
- Before any line-based insertion: read ~5 lines on each side of the target to confirm it sits between top-level definitions, not inside a body.
- After any mid-file insertion: search the file for displaced/duplicate fragments and remove them.
<!-- END SERENA SECTION -->
