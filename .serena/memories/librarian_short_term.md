# Librarian Short-Term Observations

## Run: 2026-08-03 (second invocation) — probe_interlobe_corridor + pearcey_table.npz

**Scope:** Post-commit sync re-invocation for the same two commits as the prior run:
- `de1aebd`: `scripts/probe_interlobe_corridor.py` + agent state/memory files
- `c715bcde`: `cogwheel/data/pearcey_table.npz` + `scripts/measure_cusp_arm_reach.py`

**Outcome:** No-op. Already handled.

The previous Librarian run already:
1. Analyzed both commits — concluded no doc surfaces were stale (scripts-only changes, no new
   disk artifacts beyond `pearcey_table.npz` which already had a DATA_CONTRACTS.yaml entry).
2. Committed the no-op sync record as `5e8c63b` ("docs: post-commit sync (probe_interlobe_corridor
   + pearcey_table.npz — no-op)").
3. Deleted `sync_issues.json`.

**Why this run was triggered:** The task was enqueued before or during the previous run's file
deletion — a race condition in the post-commit hook queue. Safe to ignore on re-entry when
`sync_issues.json` is already gone and the no-op commit is already in the log.

**Pattern to watch:**
- When a Librarian run commits a "no-op" sync and deletes sync_issues.json, a second invocation
  can still arrive in the queue. The guard is: check that sync_issues.json is MISSING and that
  the log already contains a post-commit sync commit covering those hashes — if both are true,
  the run is a confirmed duplicate and requires no action beyond a memory write.

**Fragile cross-references (carried forward from previous run):**
- `pearcey_table.npz` is shipped as package data (`cogwheel/data/`). DATA_CONTRACTS.yaml entry
  and data_registry.yaml entry are present and correct. If the file is regenerated with a
  different schema or path, both entries need updating simultaneously.
- If `_DEFAULT_TABLE_NAME` in `_pearcey_table.py` changes, DATA_CONTRACTS.yaml's description
  and data_registry.yaml's `relative_path` both need updating.
