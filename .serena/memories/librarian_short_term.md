# Librarian Short-Term Observations

## 2026-08-11 -- interior cusp serving barrier post-commit sync (no-op)

Scope: commit `a8361be feat(lensing): serve interior cusp sources via
calibration bypass + ppGO fold-band gate`. Triggered by post-commit
`.claude/sync_issues.json`.

### What happened

The previous librarian session (same date, same build) had already done ALL
the doc-sync work for this build and bundled it INTO commit `a8361be` itself:
- `changelog.d/2026-08-11_interior_cusp_serving_barrier.md` (CHANGELOG.md)
- `completed.d/2026-08-11_interior_cusp_serving_barrier.md` + `2026-08-11_mpmath_hang_fast_tier.md`
- `spec_changelog.d/2026-08-11_interior_cusp_serving_barrier.md` (SPEC 0.37.5 -> 0.37.6)
- SPEC.md surgical edit: ppGO gate enumeration + interior cusp serving sentence
- Dangling wiki-link fix in `todo.d/lensing_serving_ladder_guards_are_red.md`

Triage on the new sync_issues.json:
- `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`: internal implementation
  change only; no new modules, no API signature changes, no new disk artifacts.
  `docs/source/` has no mention of cusp/pearcey/serving in narrative pages —
  confirmed NO-OP by prior session and reconfirmed this run.
- Test files in `cogwheel/tests/` — skipped per triage rules.
- `sync_derived_docs.py`: ran clean; only recurring `lens_amplification_surrogate`
  test-only-consumer warnings (escalation TODO fragment already open — do NOT
  create a duplicate). "Some issues auto-fixed" produced zero real git diff.

### Outcome: no-op sync

### Patterns / gotchas this run

- When the feature commit BUNDLES the doc-sync work from a preceding librarian
  session, the post-commit hook fires again for that feature commit and creates
  a new sync_issues.json — but no additional work is needed. Pattern: check
  `git show --stat <hash>` to see if spec/changelog fragments are already IN
  the commit before doing any work.
- "Some issues auto-fixed" from `sync_derived_docs.py` with zero git diff is
  the recurring internal-state-flush no-op (see librarian_knowledge for this
  pattern — already documented).

### Cross-references to watch (carried forward from prior session)

- SPEC.md cites `_ETA_MAX_FOLD` twice (fold-arm fence + ppGO gate leg) — if
  a future build moves the fold-arm fence, both citations must move.
- Interior cusp serving sentence names `radius >= radius_min` and 3-vs-1
  stationary split — fragile if the interior/exterior serving rule changes
  again.
- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER STILL STALE (carried from
  2026-08-10, INS-1-002/003): SPEC.md ~line 63 and DATA_CONTRACTS.yaml ~line
  199 still describe `exterior_polar_rho_log_carrier_v1` as "the ONLY known
  tag" — stale since V5 shipped. Pending; do not close until fixed.
- Lobe axis-schema contract (INS-4-002/F050): DATA_CONTRACTS.yaml still
  describes old lobe axis schemas (raw-theta V1, sqrt-edge); production code
  ships `lobe_caustic_relative_v1`. Pending.
