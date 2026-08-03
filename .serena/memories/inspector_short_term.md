# Inspector Short-Term Observations

## Review: 2026-08-02 (pass 2) — WP1: interior_w_nodes_per_decade density lever (fix pass)

### Scope
- Production: `cogwheel/lensing/surrogate_training.py` — `interior_w_nodes_per_decade` field on `TrainingConfig` (default 15), wired into BOTH `_train_band_charts` AND `_subdivide_farfield_tile` for `'interior'` and `'lobe_interior'` regions. The `build_lobe`/`build_ff` closures now pass `w_nodes=eff_w_nodes` (the correctly dispatched value) instead of the old `tile_w_nodes` (which was None for interior tiles).
- Tests: new `InteriorWnpdAccuracyTestCase` + `TrainingConfigWnpdFieldTestCase` in `cogwheel/tests/test_lensing_exterior_windows.py` (8 tests pass, ~38 s).
- Existing suite: all 39 (non-skipped) tests in `test_lensing_surrogate_training.py` pass unchanged.
- SPEC.md: unrelated Born-rung and ghost-decay-gate text updated; `interior_w_nodes_per_decade` not explicitly named in SPEC (Librarian-scope note, not a finding).

### Findings
None. The build is clean.

### Previously Open Findings — Re-checked

1. **INS-2-001** (subdivision fallback for interior w-density): **RESOLVED.** Lines 3655-3666 now correctly dispatch `elif region in ('interior', 'lobe_interior'): eff_w_nodes = config.interior_w_nodes_per_decade` in `_subdivide_farfield_tile`, mirroring the main tiler path at lines 4330-4335. Verified by reading the actual code, confirming `region` is available from line 3652, and confirming the test suite passes.

2. **INS-1-001** (unreachable `C <= 0.0` guard in ppgo_map.py): STILL PRESENT. Trivial.
3. **INS-1-002** (DATA_CONTRACTS empty-range semantics): STILL PRESENT. Trivial / Librarian scope.
4. **INS-1-003** (misleading `_EXTRAP_W_CERT_DEFLATION` name): STILL PRESENT. Trivial.

### Correctness Assessment
- The three-branch dispatch (`w_nodes is not None` / `region in interior` / else) is now CONSISTENT between `_subdivide_farfield_tile` (line 3660) and `_train_band_charts` (line 4330). Both paths use the same pattern.
- `build_lobe` and `build_ff` closures capture `w_nodes=eff_w_nodes` instead of `tile_w_nodes`. This is correct: for interior tiles, `tile_w_nodes` was None (the dict has no `'w_nodes_per_decade'` key); now the explicitly dispatched value is passed.
- The `_build_farfield_chart` and `_build_lobe_chart` functions' own internal fallback (`config.w_nodes_per_decade` when `w_nodes_per_decade is None`) never triggers for tiler-dispatched calls because the tiler now always passes an explicit value.
- TrainingConfig addition is backward-compatible: frozen dataclass, new field has default (15), existing callers unaffected.
- The test `test_w_node_count_changes_with_wnpd` hardcodes expected node counts (33, 17) for a ~2.6-decade band at 12 and 6 nodes/decade — arithmetic checks out: ceil(2.6 * 12) + 2 = 33, ceil(2.6 * 6) + 2 = 17 (assuming the _log_w_grid formula's linspace semantics).
- No stale references to the old fallback pattern remain in production code.

### Open Issues Carried Forward
- INS-1-001, INS-1-002, INS-1-003 (all trivial/Librarian scope)
