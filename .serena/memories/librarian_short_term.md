# Librarian Short-Term Observations

## Run: 2026-08-03 — probe_interlobe_corridor + pearcey_table.npz commits

**Scope:** Post-commit sync for two commits:
- `de1aebd`: added `scripts/probe_interlobe_corridor.py` + agent state/memory files
- `c715bcde`: added `cogwheel/data/pearcey_table.npz` + `scripts/measure_cusp_arm_reach.py`

**What went stale and why:**

Nothing. This was a genuine no-op sync.

**Analysis:**

1. `scripts/probe_interlobe_corridor.py` — standalone diagnostic probe under `scripts/`. Prints
   to stdout only; writes no disk artifacts. Not under `cogwheel/`, so does not affect module
   lists, API docs, or public API. No DATA_CONTRACTS.yaml entry needed.

2. `scripts/measure_cusp_arm_reach.py` — standalone measurement script under `scripts/`. Prints
   to stdout only; writes no disk artifacts. Uses `use_pearcey_table()` to load the table, but
   does not produce any new disk outputs.

3. `cogwheel/data/pearcey_table.npz` — shipped package-data artifact. DATA_CONTRACTS.yaml
   already has a complete `pearcey_table` entry (lines ~214-224) with format, producer
   (`scripts/train_pearcey_table.py`), and consumers (`_pearcey_cusp.py: cusp_amplification`
   and `use_pearcey_table`). `data_registry.yaml` also has a corresponding entry. No new entries
   needed.

**sync_derived_docs.py Step 0 result:**
- Ran cleanly. `git diff` empty after script (internal state flush, not a real diff).
- Consumer graph warnings for `lens_amplification_surrogate` test-file consumers are expected
  (test-file-only callers excluded from contract consumer list by convention).

**Fixes applied:** None.

**Fragile cross-references to watch:**
- `pearcey_table.npz` is now shipped as package data (`cogwheel/data/`). DATA_CONTRACTS.yaml
  entry and data_registry.yaml entry are both present and correct. If the file is regenerated
  with a different schema or path, both entries need updating simultaneously.
- If `_DEFAULT_TABLE_NAME` in `_pearcey_table.py` changes, DATA_CONTRACTS.yaml's description
  and data_registry.yaml's `relative_path` both need updating.

**Surprises:**
- The sync_issues.json listed `c715bcde` (pearcey_table.npz commit) as a second pending commit
  — the caller's task description only mentioned `de1aebd`. Always check all commits in
  sync_issues.json, not just the one named in the task preamble.
- Both DATA_CONTRACTS.yaml and data_registry.yaml were already populated for the pearcey_table
  artifact ahead of the file being committed (written in anticipation during an earlier build).
  This is the correct pattern — contract before artifact.
