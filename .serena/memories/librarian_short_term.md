# Librarian Short-Term Observations

## 2026-08-17 (post-commit sync #2: census band-ladder fix batch, commit b044c89)

- Verified 5990a8d's SPEC.md edit (Microlensing-engine row) BY DIFF against
  actual code (`SERVE_ROUTES` in `cogwheel/lensing/serve_route_census.py`):
  the 8-label MECE set (surrogate/ppgo_above_ceiling/saddle_c3/born_analytic/
  analytics_engine_hosted/engine_residual/wave_refused/engine_refused) and
  the corrected 10k demand numbers match the shipped constant and
  `_classify_nodes` routing logic. No further edit needed — driver's SPEC
  update was accurate.
- F083 (FINDINGS.md), lensing_tube_angular_axis_graduation.md,
  lensing_saddle_c3_band_split_serving.md, sdk_escalation_wait_starves_
  watchdog.md all render cleanly into TODO.md/FINDINGS.md with the existing
  render_fragments.py pass — no dangling-link or rendering issues beyond the
  5 pre-existing FINDINGS-link dangles (unrelated, tracked, out of Librarian
  scope per the "THIRD OCCURRENCE GETS A FRAGMENT" note).
- CLOSED the `lens_amplification_surrogate` consumer_graph advisory for
  `scripts/serve_route_census.py::main` (--with-artifact mode via
  `LensAmplificationSurrogate.load`) — this is a genuine PRODUCTION
  consumer, not test-only, so tagged `kind: script` (new kind value; only
  `kind: test` existed before). `kind` is inert to the checker regardless
  (matches purely on module+function) so `kind: script` is purely
  documentation, same mechanism as the established `kind: test` pattern.
  Added `contracts_changelog.d/2026-08-17_serve_route_census_script_
  consumer.md` (bump: minor, no `date:` field — matches the established
  2026-08-13/15 test-consumer-registration fragments' convention, renders
  into the harmless empty-date bucket).
- completed.d/2026-08-17_serve_route_census.md quoted the OLD (pre-audit)
  83.37%/7-label breakdown as its measured HEAD-report claim. Per the
  "historical claim stays as measured" convention, did NOT edit that
  number — appended a new "CORRECTION (2026-08-17, commit 5990a8d)"
  paragraph at the end of the fragment pointing to the corrected 10k JSON
  and the new 72.25%/8-label numbers, explicit that the old number is
  superseded-not-wrong-for-its-own-code-state.
- NEW OBSERVATION: a Librarian doc-only sync commit that stages ONLY
  `.claude/spec/*` files (COMPLETED.md, DATA_CONTRACTS.yaml/_CHANGELOG.md,
  a completed.d fragment, a new contracts_changelog.d fragment) — with
  NEITHER `.serena/memories/librarian_short_term.md` NOR
  `.claude/sync_issues.json` staged in that same commit — does NOT match
  the pre-commit hook's librarian-sync-commit fingerprint (line ~175:
  `^\.serena/memories/librarian_(short_term|knowledge)\.md$|^\.claude/
  sync_issues\.json$`). A post-commit mechanism then regenerates a FRESH
  `.claude/sync_issues.json` pointing right back at the sync commit's own
  changed files — self-referential, since those files were already synced
  by the commit that produced them. Deleted it as a legitimate no-op
  (nothing further to sync). If this recurs, consider staging the
  short-term memory write in the SAME commit as the doc sync to get the
  hook's exemption and avoid the self-referential regeneration — flag for
  a future session rather than fixing now (mechanism lives outside
  Librarian's `.claude/spec/`+`cogwheel/`+docs edit scope).
- Reverted a stray `.claude/tidy_advisory.json` diff left by
  `render_fragments.py` (pre-existing at session start, unrelated to this
  batch) before staging/committing — per the standing rule, never commit
  that side effect.
- Left untouched (per POST-COMMIT MODE scope fence): `cogwheel/lensing/
  surrogate.py`, `cogwheel/lensing/surrogate_training.py` (owned by the
  in-flight `tube_angular_axis_graduation` SDK build), and the
  architect/coder/professor agent_state + short_term memory files
  (foreign concurrent agent state, not mine to stage).
