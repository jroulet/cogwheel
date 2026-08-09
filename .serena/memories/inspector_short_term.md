# Inspector Short-Term Observations — 2026-08-08

## Review scope
ExteriorPolarChart cusp-adapted u = d^(2/3) coordinate (Brief: exterior_polar_cusp_coordinate). Full review of all uncommitted changes across 10 test files + 2 production files. Re-check of INS-3-003 and INS-3-004.

## Resolved findings
- **INS-3-003 (RESOLVED)**: `_exterior_cusp_axis_map` helper in `test_lensing_farfield_envelope.py` now mirrors production origin logic (waist-based origin selection, null fallback for unrepresentable tiles). Both `_train_tile` and `_train_exterior_chart` call it. The `@_TRAIN_TIER_SKIP`-masked crash is fixed. All farfield tests pass (36 passed, 17 skipped).
- **INS-3-004 (RESOLVED)**: `_synthetic_exterior_polar_chart` sentinel block rewritten with explicit 3-case contract: (1) both omitted → identity-like map; (2) either None → both None (raw-theta fallback); (3) exactly one real value → ValueError. `SentinelAxisContractTestCase` pins all three branches. `SentinelAxisContractSelfFalsification` proves sentinel leakage is rejected.

## New findings

### BUG INS-4-001: Wedge loader `.get()` breaks V3 required-key contract
**File**: `cogwheel/lensing/surrogate.py`, `_chart_from_npz` wedge branch (~line 4094)
**Severity**: bug
**Description**: The coder changed the wedge loader from `data[prefix + 'theta_to_u']` (hard KeyError) to `data.get(prefix + 'theta_to_u')` (None fallback). The InteriorWedgeChart V3 schema REQUIRES theta_to_u (from_wedge_engine always builds it; `_chart_to_npz` always writes it). The soft fallback breaks the contract: a corrupt V3 artifact missing theta_to_u would silently load with theta_to_u=None and potentially interpolate on wrong axes. Test `test_v3_missing_theta_to_u_raises_keyerror` in `test_lensing_wedge_dd_arclength.py:901` now FAILS.
**Suggested fix**: Revert the wedge loader line back to `data[prefix + 'theta_to_u']` (hard KeyError). The `.get()` fallback is correct for the exterior-polar loader (new optional field) and acceptable for the lobe loader (fixing a pre-existing latent trap where theta_to_u=None charts couldn't round-trip), but WRONG for the wedge loader. Change wedge line only: `theta_to_u = data[prefix + 'theta_to_u']`.

### BUG INS-4-002: Unskipped test classes use invalid envelope definition tag
**File**: `cogwheel/tests/test_lensing_exterior_windows.py`, lines 2292-2890
**Severity**: bug
**Description**: Three test classes (`WholeInteriorSacrcTestCase`, `WholeInteriorSacrcLiteralBarTestCase`, `InteriorWnpdAccuracyTestCase`) were unskipped from `@unittest.skip("Polar re-chart: fixture needs ExteriorPolarChart migration")` but all use `definition=ch.INTERIOR_SACR_C` (='interior_sacr_c_envelope') with `LensAmplificationSurrogate.from_engine`, which validates the tag against `KNOWN_FARFIELD_DEFINITIONS` via `_validate_farfield_definition`. The SACR-C interior tag is NOT in the far-field set, so every test raises ValueError. The tests were correctly skipped before and the unskip was premature — they need a different API path (or `from_engine` needs to accept non-farfield definitions).
**Suggested fix**: Either re-apply the `@unittest.skip` decorators, or add proper support for non-farfield definition tags in `LensAmplificationSurrogate.from_engine`. Since the definition tag validation (`_validate_farfield_definition`) is pre-existing and intentional (from_engine trains exterior far-field charts), the skip is the safer immediate fix. A more complete fix would route through a different training API or widen the definition validation.

### DESIGN INS-4-003: SPEC.md still references old axis schema tag
**File**: `.claude/spec/SPEC.md`, line ~62
**Severity**: design
**Description**: SPEC.md still says `axis-schema tag 'exterior_polar_rho_theta_c'` and "no arc-length map is needed". Code now uses `'exterior_polar_rho_u_v1'` and has optional `theta_to_u`. (INS-3-001 carried forward, Librarian scope.)

### DESIGN INS-4-004: DATA_CONTRACTS.yaml still describes old schema
**File**: `.claude/spec/DATA_CONTRACTS.yaml`, line ~198
**Severity**: design
**Description**: DATA_CONTRACTS.yaml still describes `axis_schema='exterior_polar_rho_theta_c'` and missing `theta_to_u` field for ExteriorPolarChart. (INS-3-002 carried forward, Librarian scope.)

## Pre-existing issues
- None new carried forward.

## Test suite status (fast tier)
- `test_lensing_surrogate.py`: 84 passed
- `test_lensing_exterior_windows.py`: 53 passed, 1 failed (INS-4-002), 4 errors (same cause)
- `test_lensing_farfield_envelope.py`: 36 passed, 17 skipped
- `test_lensing_exterior_polar_fold.py`: 26 passed
- `test_lensing_surrogate_census.py`: 19 passed, 14 skipped
- `test_lensing_wedge_dd_arclength.py`: 30 passed, 6 skipped, 1 failed (INS-4-001)
- `test_lensing_surrogate_lobe.py`: 63 passed, 10 skipped
- `test_lensing_surrogate_training.py`: not run (gated behind COGWHEEL_TRAIN_TIER)
