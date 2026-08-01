2026-08-01 Build 1e-lobe review (pass 3): PASS.

## Scope
WP1: Lobe s-coordinate (sqrt-edge) production code in `cogwheel/lensing/surrogate.py` + new acceptance tests in `cogwheel/tests/test_lensing_surrogate_lobe.py`.

## Findings
No blocking issues. Two previously-identified Librarian-scope documentation divergences remain open (INS-1-001, INS-1-002) — re-verified as still valid (neither SPEC.md nor DATA_CONTRACTS.yaml was modified in this build).

## Verified (pass 3 — re-run against same diff)
- Math of `s = sqrt(span) - sqrt(theta_max - theta)` coordinate transform is correct (forward and inverse verified algebraically; endpoints land exactly via forced assignment; closed-form oracle in tests is non-circular by inspection).
- `from_lobe_engine` constructs the sqrt-edge map correctly: `theta_fine` linspace, `s_fine = s_total - sqrt(theta_max - theta_fine)`, nodes placed as images of uniform s, endpoints forced.
- `_evaluate_chart` correctly branches on `chart.theta_to_s is not None` — V1 (None) path uses raw `theta_local` directly, new path maps through the stored 2001-node map via `np.interp`.
- Serialization: `_chart_to_npz` writes `theta_to_s` only when not None, stamps V1 vs current schema tag appropriately; `_chart_from_npz` requires the key only for the current schema, tolerates absence for V1.
- `from_lobe_values` validates both `theta_to_s` and `s_grid` or neither (loud ValueError on mismatch).
- `_assemble` passes `theta_to_s` through and the dataclass stores it as a frozen field.
- `_lobe_serves` is independent of `theta_to_s` (operates on theta for containment).
- Caller/callee consistency: all callers of `from_lobe_values` and `_assemble` pass the new optional args correctly (checked via `find_referencing_symbols`).
- All test helper callers (`_build_lobe_chart`, `_build_v1_lobe_chart`, `_build_uniform_lobe_chart_at_shifted_range`) pass `theta_to_s`/`s_grid` args consistently.
- 54 lobe tests pass; 69 main surrogate tests pass; 31+49s training tests pass; 14+13s census tests pass.
- Import clean (`import cogwheel.lensing.surrogate` succeeds).
- `_NODE_EXACT_TOL` 1e-10 → 1e-7 justified by the theta→s interp budget (~6e-9) at 2001 nodes; commented justification is sound.
- New `_LOBE_ARC_MAP_SIZE = 2001` constant parallels existing `_FARFIELD_ARC_MAP_SIZE`.
- V1 identity-path tests cover: serve returns finite, schema tag round-trips as V1, theta_to_s key absent in npz, served values byte-identical before/after save-load, mutation detection (indexing None crashes).
- sqrt-edge acceptance tests cover: closed-form oracle match, round-trip budget, strict monotonicity, shape check, bound-shift stability, no-worse-than-uniform, self-falsification.
- Spline axis order verified: knots encode s-domain (from fit on `spline_axes=(log_w, gamma, rho_lobe, s_grid)`); at serve time theta→s interp feeds s to the spline's fourth axis. Consistent.
- `_validate_lobe_axis_schema` validates against frozenset({V1, current}); unknown tags hard-refuse.

## Open issues (carried forward — Librarian scope)
- INS-1-001: DATA_CONTRACTS.yaml describes theta_to_s only for TubeChart; LobeInteriorChart now also serializes it.
- INS-1-002: SPEC.md describes lobes as "(rho_lobe, theta_local)" with a single schema tag; code now has V1+current dual-schema with sqrt-edge s as the spline's fourth axis.
