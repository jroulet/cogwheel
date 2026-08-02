# Librarian Short-Term Observations

## Run: 2026-08-01 — Build 3 C6 + test-fix inspector pass

### Scope
Working tree vs HEAD. Code changes: `surrogate_training.py` (C6: curvature-relative
eta_max/eta_floor), `surrogate.py` (lobe sqrt-edge coordinate already covered by
prior librarian run), test file fixes (test-only; skipped per rules).

### What went stale and why
1. **SPEC.md FOOT-OF-NORMAL sentence** — said "skips a tube chart whose eta_max
   exceeds half R_c". C6 replaced the skip with `assert f_max < 0.5`, making it
   vacuous. The coder committed the C6 code but left SPEC.md describing the OLD
   skip-and-record behavior. Pattern: implementation-level guard changes often
   land in code without touching SPEC.md.

2. **`lensing_caustic_relative_coordinates.md` step 3 (C6)** — still listed as
   open pending item. Needed DONE marker.

3. **`lensing_coverage_map.md` row 3** — cited "Two bands SKIPPED by foot-of-normal
   guard because eta_max=0.05 is ABSOLUTE". C6 makes this false. Updated to
   "foot-of-normal cause CLOSED (C6 2026-08-01); only topology-sliver cause remains."

### Fragments created
- `spec_changelog.d/2026-08-01_c6_curvature_relative_tube.md` (bump: patch)
- `completed.d/2026-08-01_c6_curvature_relative_tube.md`

### SPEC_CHANGELOG ordering note
`2026-08-01_lobe_sqrtedge_coordinate.md` sorts alphabetically after
`2026-08-01_c6_curvature_relative_tube.md`, so the lobe fragment gets 0.31.2
and C6 gets 0.31.1 — inverse chronological appearance in SPEC_CHANGELOG.md.
Expected render quirk; don't fix.

### sync_derived_docs.py output
4 test-file-only consumer warnings for `lens_amplification_surrogate` via
`LensAmplificationSurrogate.load`. Per convention, test-file callers stay off
DATA_CONTRACTS.yaml. The script reported "5 checks run, some issues auto-fixed"
but git diff showed no doc-file changes from the script — the "auto-fix" was
likely a no-op internal state flush.

### Fragile cross-references to watch
- SPEC.md now cites C6 (`eta_max = f_max * R_c`) by mechanism; if `f_max` /
  `f_floor` are renamed again or the formula changes, SPEC.md needs updating.
- `lensing_coverage_map.md` row 3 still shows OPEN (1b not done); don't mark
  it closed until step 1b ships.
- The `lensing_caustic_relative_coordinates.md` fragment now has C6 done but
  steps 1b, 4, 5, 6, 7, 8, 9 still open; the fragment itself stays in todo.d.
