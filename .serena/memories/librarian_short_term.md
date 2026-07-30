## 2026-07-30 post-commit sync (--post-commit c7e15f2)

Scope: 7 pending commits from `.claude/sync_issues.json` (b92f413 through
c7e15f2 — the F043/F044/F045 HEAD-oracle-guard aftermath + build 1d
wedge/tube-normal follow-ups + the SDK teardown-result fix). Outcome:
**verify-only, zero doc edits** — every doc surface was already correctly
synced by the in-DAG Librarian run (903946a) or by the commits themselves.

Checked and found current:
- `scripts/sync_derived_docs.py` ran clean (0 diff); only its usual
  test-file-only `lens_amplification_surrogate` consumer flags fired
  (known benign, production-only convention — memory already covers this).
- FINDINGS.md has F043/F044/F045; every citation of them across
  `.claude/spec/{TODO,COMPLETED}.md`, todo.d/completed.d fragments,
  `.claude/hooks/*`, `cogwheel/lensing/chang_refsdal/geometry.py`, and the
  three lensing test files resolves.
- `todo.d/sdk_head_relative_test_guard.md` was correctly retired
  (deleted, mirrored into `completed.d/2026-07-30_sdk_head_relative_test_guard.md`);
  `todo.d/tests_head_oracle_sweep.md` correctly stays OPEN (multi-part
  sweep, only 3 of ~10 call sites done per its own body).
- `df3770`'s two new todo.d fragments (lensing_caustic_relative_coordinates,
  lensing_collocation_from_local_scales) are present in the rendered
  TODO.md.
- `0b99a3a`'s geometry.py change (LensDomainError message: wedge edge is
  NOT "the deltoid cusp", per F044) is error-message prose only, no
  signature change. Grepped "deltoid cusp" repo-wide: the only place it
  still describes the wedge edge is FINDINGS.md's own F044 text, which
  QUOTES the old wrong claim to refute it — correct, not stale. No
  SPEC.md/docs/source reference to fix.
- SPEC.md has no mention of `_tube_normal`/`_WEDGE_EPS` (implementation
  detail, correctly left out per the "SPEC carries architecture, not
  perf/impl detail" pattern) and no docs/source page mentions HEAD-relative
  tests (agent-only concern, correctly absent).
- `.claude/sdk/*`, `.claude/hooks/*`, `.claude/handoff/*`,
  `.serena/memories/*`, `.claude/tidy_advisory.json`,
  `.claude/agent_state/*` in the changed-files union are agent-only paths
  outside Librarian scope (per CLAUDE.md sync-to-main exclusions + the
  Librarian's read-only boundary on other agents' memories) — correctly
  skipped, not silently missed.
- `cogwheel/tests/test_guard_probe.py` and
  `test_lensing_surrogate_training.py` changes are test-only — skipped
  per the standard triage table.

Pattern worth flagging forward: this is the SECOND consecutive post-commit
run (after cece16b/1c) where the in-DAG Librarian commit already closed
out the bulk of the sync, leaving only a verify pass for the driver's
follow-up commits. When the driver's task message names which commits are
"already synced by the in-DAG run" vs "driver follow-ups," trust but still
independently grep — this run confirmed the driver's framing was accurate,
but per long-term memory that verification is never skippable.
