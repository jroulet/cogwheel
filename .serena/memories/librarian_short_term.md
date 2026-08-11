# Librarian Short-Term Observations

## 2026-08-11 -- post-commit sync for e1158cc (NO-OP)

Scope: docs: todo + brief for zero-quadrature Pearcey hot path build.

Changed files:
- `.claude/handoff/brief_zero_quadrature_pearcey.md` — build brief (agent-only path)
- `.claude/spec/TODO.md` — already regenerated in the commit itself
- `.claude/spec/todo.d/lensing_zero_quadrature_pearcey.md` — source fragment

No `cogwheel/` Python files changed. No new modules, no new disk artifacts, no public API
changes, no Sphinx doc updates needed. TODO.md was already rendered as part of the commit.

POST-COMMIT SYNC NO-OP RULE applies: agent/spec-only commit with no downstream doc
surface impact. "Record no-op sync runs as a commit rather than skipping silently" — sync
commit carries only the sync_issues.json deletion + this memory write.

Pre-existing dirty files `.claude/agent_state/librarian.json` and
`.claude/tidy_advisory.json` were present before this run (render_fragments.py side
effects from a prior session) — not committed.

## Previous session carry-forwards (still pending):

- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER (INS-1-002/003): SPEC.md ~line 63
  and DATA_CONTRACTS.yaml ~line 199 still describe
  exterior_polar_rho_log_carrier_v1 as "the ONLY known tag" -- stale since
  V5 2D tag shipped. Both surfaces need updating. Still pending.
- Lobe axis-schema DATA_CONTRACTS.yaml rows (INS-4-002/F050) deferred.
- lensing_farfield_sd_coordinate_degenerates + name_spans_three_regimes open.
- surrogate_contract_test_consumer_warning escalation fragment open; no dup.
