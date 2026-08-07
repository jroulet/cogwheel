# Librarian Short-Term Observations

## 2026-08-07 wedge probe v3 + m_lens_range post-commit sync

**Scope**: Two pending commits from sync_issues.json:
- `6144378` — `feat(lensing): m_lens_range override for train()/PriorBox` (surrogate_training.py)
- `a996316` — `scripts: wedge probe to single mass stratum (v2-equivalent scope)`

**Findings**:
- `6144378` was already fully processed by the previous librarian session (commit `c3277b6`
  "docs: post-commit sync (m_lens_range — all surfaces current, no-op)"). That commit
  updated the short-term memory but NOT `librarian.json` (still showed `b1b4570`).
- `a996316` is a one-line parameter addition (`m_lens_range=(10.0, 15.8)`) to
  `scripts/probe_wedge_v3.py`. Scripts-only, no new disk artifacts, no cogwheel/ public
  API change. SCRIPTS/ REWRITE NO-OP RULE applies — no doc surfaces affected.
- `sync_derived_docs.py` ran cleanly. Only stray diff was `tidy_advisory.json` (reverted).
  Consumer-graph warnings for `lens_amplification_surrogate` test-only callers recurred
  (5th+ time) — already escalated via
  `todo.d/surrogate_contract_test_consumer_warning.md`, no further action.
- `sync_issues.json` was already deleted before this session started.
- `librarian.json` was already updated to `c3277b6` by some prior hook/process;
  just committed that update.

**Pattern noted — COMMIT TIMESTAMP ANOMALY**: `c3277b6` (post-commit sync) has author
timestamp 09:10:07, same as `6144378`, but is topologically AFTER `a996316` (09:10:59).
Git allows out-of-order timestamps; this is not a sign of corruption. The topological
chain is `6144378 → a996316 → c3277b6`.

**Pattern noted — UNCOMMITTED SCRIPT CHANGES VISIBLE**: `scripts/probe_wedge_v3.py`
had additional uncommitted working-tree changes (eps_values loop rewrite) beyond the
committed `a996316` content. These are driver work-in-progress. Per convention: not
staged, not committed by Librarian.

**Fragile cross-reference watch**: nothing new this run.
