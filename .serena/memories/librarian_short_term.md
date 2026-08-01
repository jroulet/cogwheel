## 2026-07-31 post-commit sync (--post-commit c8f6bf6, verify-only)

Scope: 22 pending commits from `.claude/sync_issues.json` (f4f49b6 through
714af39 — F052-F059 SDK/watchdog/timing fixes, the analytic caustic-reach
build (3785ccc), the 1e-farfield (s,d) port and its two cleanup follow-ups
(7d0f196, 8dc3b6f, 57b0581), three findings promoted to TODO items
(29f6571), the OpenCode spoke infra add (3621d35), and a run of `.claude/
sdk/` orchestrator/launcher/watchdog fixes). Outcome: **verify-only, zero
doc edits** — independently re-verified via `git log --oneline f4f49b6^..
714af39` and per-commit `git show --stat`, not by trusting the driver's
"bulk is infra" framing at face value.

Why zero edits, checked not assumed:
- `git diff --stat f4f49b6..714af39 -- docs/source pyproject.toml
  environment.yaml docs/requirements.txt` is EMPTY — nothing in the whole
  backlog touched a Sphinx surface or a dependency file.
- The only two commits with `cogwheel/**/*.py` changes that also carry
  spec/contract updates (3785ccc analytic-caustic-reach, 7d0f196 farfield
  (s,d) port) already ship their own `SPEC.md`/`DATA_CONTRACTS.yaml`/
  changelog fragments/`completed.d` INSIDE the same commit (driver-
  committed docs alongside code, per the now-familiar in-DAG pattern) —
  confirmed by reading those commits' own `--stat`, not just the JSON's
  changed_files list.
- The two follow-up commits (8dc3b6f: delete dead private
  `_union_cusp_nodes` + cusp-plotting test helper + restore a lost
  `lru_cache`; 57b0581: fix stale `(rho, theta_c)` axis comments in
  `surrogate_training.py` docstrings + port two gated-tier test fixture
  files) touch no public API and no docs-facing text — grepped `.claude/
  spec/` and `docs/source/` for `_union_cusp_nodes`, `FromEngineCusp
  WiringTestCase`, `_plot_cusp_nodes_on_rays`, `_pos_raw_out`: zero hits
  everywhere, so there was never a dangling reference to begin with (the
  removed function was private/dead, not documented anywhere).
- `(rho, theta_c)` IS still mentioned in SPEC.md/DATA_CONTRACTS.yaml/TODO.md
  — read those sentences and they already say "retained only for tile-
  proposal/admission", matching 57b0581's commit message ("they're now
  tile-proposal coordinates, not chart axes") exactly. Already correct,
  not stale.
- FINDINGS.md: grepped `^## F0(5[2-9]|6[0-9])` — F052 through F059 all
  present with proper headers and dates, matching every F-number cited in
  the 22 commit subjects (F052 corrected variant included). No gaps.
- Rest of the backlog (`.claude/sdk/*`, `.codex/*`, `.opencode/*`,
  `.agents/skills/*`, `AGENTS.md`, `scripts/render_fragments.py`,
  `scripts/sync_to_main.sh`, `scripts/verify_installation.sh`,
  `.claude/handoff/*`, `.claude/agent_state/*`) is agent-only infra per
  CLAUDE.md's `EXCLUDE_PATHS` / outside every row of the Librarian triage
  table — correctly out of scope, not silently skipped.
- `scripts/sync_derived_docs.py` (via `cogwheel-newlal` python, per
  standing note) ran clean: 0 tracked diff. It re-flagged the SAME 4
  test-file-only `lens_amplification_surrogate` consumer callers
  (`SerializationMultiChartTestCase`/`SerializationTestCase` round-trip
  tests) as before — same known-benign pattern, production-only consumer
  lists by convention, not a new gap.
- `scripts/render_fragments.py` reported "All surfaces up to date." with 0
  tracked diff. One stray untracked `.claude/tidy_advisory.json` appeared
  as a side effect (same intermittent pattern noted in prior runs) — left
  alone, untracked, not staged/committed.
- Two files were staged in the tree but explicitly NOT mine to touch:
  `.claude/sdk/gates.py` and `.claude/sdk/tests/test_gate_heartbeat.py`
  (driver's own in-flight change) — left staged and untouched, not part of
  this commit.
- This memory write is the only content change; committed alone (matches
  the 0756b47 precedent: a verify-only post-commit sync is still worth a
  commit for the audit trail, not a silent no-op).

Pattern worth flagging forward: FIFTH consecutive post-commit run that is
pure verify for the cogwheel-package/docs surface specifically, though this
one had the largest commit count (22) of any run so far, almost entirely
`.claude/sdk/` build-launcher/provider-routing infra (Codex/OpenCode spoke
work) plus two small lensing cleanup commits riding on an already-synced
port. Large commit COUNT does not imply large doc-sync WORK — always
compute the actual touched-surface diff before assuming otherwise.
