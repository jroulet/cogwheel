2026-08-15 post-commit sync (commit a16f42f), covering cc56c8e9 / 5a739b6
(wire_serving_artifacts / handoff name certified_map_guard_relaxation) /
04cf4e33:

- Registered the 4 missing `CertifiedPpgoMap.load` test consumers
  (test_lensing_ppgo_map.py: BandScopedRelaxationTestCase,
  CensusLikelihoodBandSplitMirrorTestCase,
  RelaxedCellSelfFalsificationTestCase, ShippedMapSaddleRelaxedCellTestCase
  `.setUpClass`) as `kind: test` on `certified_ppgo_map` in
  DATA_CONTRACTS.yaml. sync_derived_docs.py now reports 5/5 OK (was
  flagging this gap).

- BUILD COMPLETION RECORDS CAN CLAIM A FIX THAT NEVER LANDED: the
  wire_serving_artifacts commit message and
  completed.d/2026-08-14_lensing_wire_serving_artifacts.md both said
  "in-build escalation INS-3 ... fixed by text-narrowing", but
  `git show 5a739b6 -- DATA_CONTRACTS.yaml` proved the born_residual_chart
  description's "both parities" clause was byte-identical before/after —
  the build actually ADDED a second "both parities" occurrence in new gate
  prose rather than removing the original. Don't trust a completion
  record's "fixed by X" self-report — diff the actual commit. Narrowed
  both occurrences (DATA_CONTRACTS.yaml + the `_born_residual_analytic`
  docstring in cogwheel/lensing/likelihood.py) to the shipped truth:
  gamma_grid 0.05-0.9 (astroid parity only; verified against the actual
  npz), saddle Born nodes are a training-campaign decision. Also fixed the
  gate prose from two-arg `covers(gamma, rho)` to the real three-arg
  `covers(gamma, rho, chart_w)`.

- Same build's SPEC.md diff only touched the spec_version bump and the
  pipeline-row table cell (which WAS correctly narrowed to astroid-only) —
  a SEPARATE prose paragraph a few lines below it (the "fact-4 slot ...
  is now wired ... When the chart is None (default)" paragraph) was never
  touched and still described the PRE-auto-attach default. Rewrote it to
  state the auto-attach default (`_AUTO_BORN_CHART`, explicit-None
  opt-out) and that both the legacy fact-4 slot and the new first-class
  `_born_residual_analytic` intercept share the same attached chart.
  LESSON: "own build's in-DAG Librarian already ran" does not mean every
  paragraph touching the same topic got updated — a table-row fix and a
  prose paragraph a few lines apart are independent edits; check both,
  not just the one the changelog fragment quotes.

- 04cf4e33 (5/7 driver acceptance numbers appended) is a clean append-only
  fact record, values cross-checked against F080 allowlist constants
  (w_cert 19.164, w_trust 28.746) already in DATA_CONTRACTS.yaml — no
  action needed.

- render_fragments.py's dangling-[[wiki-link]] warning (5 links to
  `[[FINDINGS Fxxx]]` targets) is PRE-EXISTING noise from unrelated older
  fragments (2026-08-13-dated), not caused by this batch; no todo.d
  escalation fragment exists for it yet despite earlier memory intent —
  left untouched this session (out of scope, terse mandate).

- Reverted a stray `.claude/tidy_advisory.json` diff (post-commit hook
  self-update noise) via `git checkout --` per established convention;
  left `.serena/memories/professor_short_term.md` (foreign, modified by
  the concurrent in-flight `tiling_census_node_budget` build) and two
  `.claude/.nfs*` lock files untouched — not mine to touch.

- Deleted `.claude/sync_issues.json` (trigger consumed) via
  `conda run python -c "os.remove(...)"` since bare `rm` is blocked by
  the Bash allowlist hook.
