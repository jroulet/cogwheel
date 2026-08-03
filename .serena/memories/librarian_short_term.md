# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit audit after fold-corrected ppGO and prior builds

### Scope

sync_issues.json covered 25 commits from 2026-08-01 to 2026-08-03, spanning:
- SDK-only changes (gates.py, runtime_*.py, orchestrator.py) — agent-only paths, skipped
- cogwheel/lensing: _airy_fold.py (fold_ppgo_correction added), channels.py
  (born_carrier_from_partition, ghost decay gate wiring), ppgo_map.py (envelope
  extrapolation fallback + smoothing), surrogate.py + surrogate_training.py
  (interior SACR-C, w_nodes_per_decade), surrogate_census.py
- Test files only: skipped per triage rules

### What went stale and the pattern

**SPEC.md Born exterior rung label in description body** (auto-fixed by sync_derived_docs.py):
The previous librarian run (2026-08-02) fixed the OPENING SUMMARY PHRASE
("Born exterior rung carrier") but apparently did NOT commit it (the
fragment `2026-08-02_c8-born-summary-label.md` was still untracked and
the working tree still had "Born far-annulus carrier" in the body cell).
`sync_derived_docs.py` caught and re-applied the fix on this run.
**Pattern**: always verify the prior librarian's fix is committed, not just
created. Untracked fragments + unstaged SPEC.md edits = prior run did not
commit its own changes.

### Surfaces checked and found clean

- docs/source/api.rst: no new top-level cogwheel modules or subpackages
  added in scope; `:recursive:` autosummary still covers the lensing
  package without manual entries.
- docs/source/overview.rst: pitched at public API level
  (ChangRefsdalChannels, LensedWaveformGenerator, etc.); no internal
  function names (fold_ppgo_correction, born_carrier_from_partition,
  _measure_cell) are cited there. No update needed.
- DATA_CONTRACTS.yaml: ppgo map consumer list unchanged; fold_ppgo_correction
  and born_carrier_from_partition are internal (not disk-artifact accessors).
- FINDINGS.md: no finding entries about ppGO degenerate-delay interior error
  exist that need updating (the 7% -> O(w^{-1/3}) improvement is in the
  commit message and module docstring; no corresponding F-numbered finding
  was created in prior builds).
- surrogate_training.py interior_w_nodes_per_decade=15: implementation
  constant, not SPEC architecture.
- ppgo_map.py envelope extrapolation fallback: internal _measure_cell detail.
- test_lensing_fold_ppgo_correction.py (877 lines new): significant test,
  but it certifies fold_ppgo_correction which is an internal ppGO
  certification utility. The CERTIFIED BY clause in SPEC's engine row
  was not updated (conserved the architecture-only level). Watch whether
  a future build explicitly requests SPEC mention of this test.

### Fragile cross-references to watch next run

- The previous librarian's STAGED SPEC.md changes were apparently reverted
  or overwrote by later agent commits. Always check whether prior fix was
  committed before assuming it's done.
- "residual chart pending training" status sentence in SPEC.md Born exterior
  rung section goes stale when BornResidualChart training artifact ships.
- fold_ppgo_correction wired into born_carrier_from_partition: if the carry
  path changes (different function in channels.py), the channels.py exports
  in __init__.py need updating.
