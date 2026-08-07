# Librarian Short-Term Observations

## 2026-08-07 wedge probe NPZ eps fix + completed.d update post-commit sync

**Scope**: One pending commit from sync_issues.json:
- `ab48b256` — `scripts+spec: wedge probe reads NPZ eps; record single-stratum result`

**Changed files in commit**:
- `scripts/probe_wedge_v3.py` — fixed eps collection to read from NPZ provenance on disk
  rather than in-memory chart.provenance (which loses heldout_eps after load). Scripts-only,
  reads existing NPZ files, no new disk artifacts. SCRIPTS/ REWRITE NO-OP RULE applies.
- `.claude/spec/completed.d/2026-08-07_driver_probes_exterior_wedge.md` — updated fragment
  with single-stratum wedge v3 results: 10 charts, 9/9 passing 5e-2 bar, eps 2.0e-3..1.6e-2,
  median 6.0e-3. Also explains two prior probe bugs (full-prior config + in-memory eps read).
- `.claude/spec/COMPLETED.md` — generated canonical, already updated in the commit.

**Subsequent commit `72ca31a`** (handoff: brief for exterior polar re-chart): only added
`.claude/handoff/brief_exterior_polar_rechart.md` — handoff/brief file, no doc surfaces.

**Analysis**:
- sync_derived_docs.py: lens_amplification_surrogate test-only-caller warning recurred again
  (6th+ time). Already escalated via todo.d fragment. No diff from script.
- render_fragments.py: "All surfaces up to date." No diff. COMPLETED.md is current.
- This is a no-op sync — all doc surfaces already current.

**Pattern noted — TIMESTAMP ANOMALY (af251cf)**: af251cf has author timestamp 09:10:59 but
is topologically AFTER ab48b256 (09:17:28). Confirmed again that af251cf only handled
commits up through c3277b6; ab48b256 was correctly identified as the new pending commit.

**Fragile cross-reference watch**: nothing new this run.
