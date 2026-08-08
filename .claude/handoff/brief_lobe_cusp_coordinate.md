# Build Brief: Lobe interior cusp-adapted angular coordinate

## Mission

Replace `theta_local` in `LobeInteriorChart` with `u = d^(2/3)` where `d` is angular distance to nearest deltoid cusp vertex. Fixes the `r_deltoid ~ |dtheta|^(1/3)` singularity that makes `rho_lobe = |y-centroid|/r_deltoid` singular at cusps. Same pattern as `InteriorWedgeChart`'s v3 cusp-adapted coordinate. Retires the `_LOBE_CUSP_EXCLUSION_DISTANCE` carve-out (no longer needed with a smooth coordinate).

## Work

1. **Add `_lobe_cusp_angles(gamma)`** — returns the three source-plane cusp-vertex angles for a given gamma. Use existing `_SADDLE_LOBE_CENTERS` and `geometry.r_deltoid`.

2. **Build `theta_to_u` map** per gamma stratum: `u = sign(d) * |d|^(2/3)` where `d = theta_local - cusp_angle` for nearest cusp. Reuse `_validate_theta_to_u` from the wedge path — same `u = d^(2/3)` convention, same validation.

3. **Update `LobeInteriorChart`**:
   - Replace `theta_local` axis with `u` in `from_engine`, `serve`, serialization
   - Store `theta_to_u` map (2-row: theta grid + u grid) — same as wedge
   - New axis schema: `'lobe_caustic_relative_v1'` (mirrors `'wedge_caustic_relative_v3'`)
   - `n_theta` nodes → `n_u` nodes in `u` coordinate

4. **Update training** (`surrogate_training.py`):
   - `_lobe_interior_tiles`: tile in `(rho_lobe, u)` instead of `(rho_lobe, theta_local)`
   - `_build_lobe_chart`: pass `theta_to_u` instead of raw theta grid
   - A tile at `u=0` (cusp vertex) should clear the 1e-3 bar

5. **Retire cusp carve-out**: remove `_LOBE_CUSP_EXCLUSION_DISTANCE` and its check in `_SaddleLobeAdmission.admts` — no longer needed

6. **Update tests**: migrate lobe tests to `(rho_lobe, u)` coordinate. The golden-file skip from the D2 fold fix can be revisited after golden regeneration.

## Measured facts (SHA fd84cea)
- `InteriorWedgeChart` has working `u = d^(2/3)` coordinate at `cogwheel/lensing/surrogate.py:537`
- `_validate_theta_to_u` at line 1153 validates the map
- `_WEDGE_AXIS_SCHEMA = 'wedge_caustic_relative_v3'`
- `_SADDLE_LOBE_CENTERS` defined in surrogate_training.py
- `_LOBE_CUSP_EXCLUSION_DISTANCE = 0.1` at surrogate_training.py (from saddle forensics build)

## Constraints
- Fast tests only. Follow AGENTS.md.
- Mirror the wedge v3 pattern exactly — no new coordinate conventions
- The `u = d^(2/3)` exponent is the SAME as wedge v3 (d = angular distance, not chord distance)
- Retire the carve-out only AFTER confirming tiles at u=0 pass the bar
