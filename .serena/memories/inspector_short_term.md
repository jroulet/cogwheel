# Inspector Short-Term Observations

## 2026-08-09 — Review: exterior_cusp_exclusion build (INS-14 follow-up, final)

Re-reviewed the exterior cusp-exclusion cut diff on branch `claude-dev`.

### Verified correct
- `_CUSP_EXCLUSION_DISTANCE` bumped 0.2 → 0.35 (calibrated via `scripts/measure_cusp_exclusion.py`)
- `_deltoid_cusp_source_angles`: D₂-folded saddle cusp angles, structurally independent of `_cusp_source_angles`
- `_exclude_near_cusp` gamma_band three-point check (gamma_lo, gamma_mid, gamma_hi)
- `_farfield_tiles` keyword-only cusp kwargs (backward-compatible, None defaults)
- Saddle exterior path wired with deltoid cusp exclusion + gamma_band in `_train_band_charts`
- `_farfield_exterior_tiles` (positive parity) unchanged — already had cusp exclusion, picks up bumped constant automatically
- All 12 new + 53 existing exterior_admission tests pass; 0 failures
- All existing callers of `_farfield_tiles` use old positional-only signature — backward-compatible
- Import chain verified
- No stale references to old constant (0.2) in codebase

### INS-14-001: RESOLVED
- `_exclude_near_cusp` line ~1798: `if not cusp_positions: return False` changed to `if not cusp_positions: continue`
- Three gamma loop now properly checks all band gammas before deciding

### No new findings
