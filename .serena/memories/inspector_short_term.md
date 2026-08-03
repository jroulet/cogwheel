# Inspector Short-Term Observations

## Review: 2026-08-03 (pass 4) — InteriorWedgeChart implementation review — PASS

### Scope
Re-review of uncommitted changes to `cogwheel/lensing/surrogate.py` (~812 lines added) and the untracked test file `cogwheel/tests/test_lensing_interior_wedge_chart.py` (40 tests).

### What Changed (identical to pass 3)
- New `_WedgeCausticMap` dataclass holding a precomputed `r_caustic(gamma, theta)` table.
- New `_interp_r_caustic` bilinear interpolation of the caustic-radius table.
- New `_to_wedge_fixed` / `_from_wedge_fixed` coordinate transforms (eigenframe ↔ wedge-fixed).
- New `_validate_wedge_caustic_map` validator.
- New `InteriorWedgeChart` frozen dataclass with `from_wedge_values` and `_assemble` class methods.
- New `_wedge_serves` guard function (cheapest-first gate ordering).
- Updated `select_chart` with a fourth loop for `InteriorWedgeChart` (lowest priority).
- Updated `_evaluate_chart` with wedge branch computing (r, theta_wedge) and optional theta_to_s remap.
- Updated `LensAmplificationSurrogate.__init__` to accept `InteriorWedgeChart`.
- Updated `serve` method's `definition` extraction to include `InteriorWedgeChart`.
- New `from_wedge_engine` training entry point.
- New `_build_wedge_provenance` provenance builder.
- Updated `_chart_to_npz` / `_chart_from_npz` with wedge branch for persistence.
- New `_WEDGE_AXIS_SCHEMA` tag for artifact versioning.

### Correctness Assessment
- **Coordinate math**: `_to_wedge_fixed` correctly implements D2 fold via `abs(y1)`, `abs(y2)`, `theta = atan2(|y2|, |y1|)`, `r = hypot / r_caustic`. Round-trip verified algebraically.
- **Axis ordering**: Consistent throughout. Coefficients `(log_w, gamma, r, theta_wedge)` → `_contract_tensor_spline(gamma, r, theta_wedge, log_w_query)` ✓.
- **NPZ persistence**: Correct: save axes as `(log_w, gamma, r, theta_wedge)` → load reads `(log_w_grid, gamma_grid, p1_grid=r, p2_grid=theta_wedge)` ✓.
- **select_chart dispatch**: Wedge charts have lowest priority (tube > farfield > lobe > wedge). No overlap risk.
- **Consumer chain**: `serve`, census `_chart_log_w_range`, `_chart_index`, `_is_band_edge`, `heldout_envelope_eps` all use duck typing on `chart.log_w_grid` / `chart.gamma_grid` — all present on `InteriorWedgeChart`. Census eps measurement's else-branch (non-FarField → `partition.envelope`) is correct for wedge charts.
- **Validator**: `_validate_wedge_caustic_map` checks gamma equality, theta span [0, π/2], finite positive r_table.
- **All 40 wedge tests pass** (36s).
- **All 69 existing surrogate tests pass** (2m19s).
- **All 54 lobe tests pass** (58s).
- **Import check**: `from cogwheel.lensing.surrogate import InteriorWedgeChart, _WedgeCausticMap` — OK.

### Findings

1. **INS-w2-001 (trivial, NOT RESOLVED)**: Test file `test_lensing_interior_wedge_chart.py` lines 882 and 1039 still contain stale comments stating "Since from_wedge_engine has a bug (LensAmplificationSurrogate.__init__ doesn't accept InteriorWedgeChart)". This bug is now fixed. The tests are functionally correct (they use the manual path for test isolation) but the docstring justification is misleading.

### Open Issues Carried Forward (pre-existing, not from this diff)
- INS-w-004 (design — Librarian scope): DATA_CONTRACTS.yaml does not describe InteriorWedgeChart. Still present.
- INS-w-005 (design — Librarian scope): SPEC.md does not mention InteriorWedgeChart. Still present.
- INS-1-001 (unreachable `C <= 0.0` guard in ppgo_map.py): STILL PRESENT. Trivial.
- INS-1-002 (DATA_CONTRACTS empty-range semantics): STILL PRESENT. Trivial / Librarian scope.
- INS-1-003 (misleading `_EXTRAP_W_CERT_DEFLATION` name): STILL PRESENT. Trivial.
