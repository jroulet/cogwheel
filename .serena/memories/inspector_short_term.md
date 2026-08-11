# Inspector Short-Term Observations

Date: 2026-08-10 (re-review, 3rd pass)
Scope: Build saddle_exterior_full_treatment — cusp-adapted u coordinate for saddle exterior tiles + parity-gated cusp-window Pearcey serve constant.

## Re-check of INS-1-001
**Still open.** SPEC.md lines 72-76: "macro-saddle (`parity == -1`) exterior charts interpolate on raw `theta_c` (no map)." The code now builds cusp-adapted `theta_to_u` maps for macro-saddle exterior tiles that contain a deltoid cusp ray at a boundary, via `_deltoid_cusp_axis_map` / `_build_farfield_chart` parity==-1 branch. The code is correct; the spec is stale. Flag to Librarian, not a code defect.

## New Findings
None. All code changes are correct.

### `_deltoid_cusp_axis_map` (surrogate.py)
- Mirrors `_wedge_cusp_axis_map` / `_lobe_cusp_axis_map` pattern.
- Correct 2/3 exponent for gamma-universal cusp-reach scaling.
- Straddle check returns None; boundary validation [0, pi/2] raises ValueError.
- `np.clip` in left-of-cusp branch safely guards against floating-point artifacts at the hi endpoint.
- Endpoint fix `theta_fine[0] = theta_lo`, `theta_fine[-1] = theta_hi` overwrites any rounding artifacts.

### `_build_farfield_chart` parity==-1 branch (surrogate_training.py)
- Probes deltoid cusp rays via `_deltoid_cusp_source_angles(gamma_mid, config.n_caustic_samples)` — same median-gamma approach as parity=1's waist computation.
- Boundary-only activation (`nearest == theta_lo or nearest == theta_hi`) prevents calling `_deltoid_cusp_axis_map` when the nearest cusp is interior (would straddle and return None).
- Falls through to `theta_to_u=None` when no candidates or interior cusp.

### `_tube_serves` parity gating (surrogate.py)
- `coverage = _SADDLE_CUSP_ARM_COVERAGE if chart.parity == -1 else _CUSP_ARM_COVERAGE` — correct dispatch.
- `_SADDLE_CUSP_ARM_COVERAGE = 0.0` is load-bearing (nonzero would admit queries the Pearcey arm cannot serve for saddle parity).

### Tests
- `TubeCuspWindowParityGatingTestCase` (test_lensing_surrogate.py): 11 tests covering both parities, mid-window/near-cusp/outside-window queries, shrink margin, diagnostic plot.
- `TubeCuspWindowParityGatingSelfFalsificationTestCase` (test_lensing_surrogate.py): 2 self-falsification tests with mock.patch.object swapping coverage constants.
- `SaddleCuspUCoordinateRoundTripTestCase` (test_lensing_surrogate_training.py): round-trip accuracy, monotonicity, endpoint matching, mismatched-row detection.
- `SaddleThetaToUMutationSelfFalsificationTestCase` (test_lensing_surrogate_training.py): held-out eps comparison with vs without cusp-adapted map.
- `CuspArmCoverageParityGateSelfFalsificationTestCase` (test_lensing_surrogate_training.py): synthetic tube charts with direct constant patching.
- `SaddleCuspAdaptedAccuracyTestCase` / `SaddleCuspAdaptedServingTestCase` (test_lensing_farfield_envelope.py): engine-backed accuracy tests (train-tier gated).

### Measurement script
- `scripts/measure_saddle_cusp_arm_coverage.py`: untracked file (not committed). Functional, complete, mirrors the positive-parity measurement methodology.

### Docstring updates
- `_exclude_ghost_dominated` and `_needs_fold_carrier`: updated from "Positive-parity only" to "Both parities". Correct.

### Import correctness
- All new symbols (`_deltoid_cusp_axis_map`, `_SADDLE_CUSP_ARM_COVERAGE`, `_deltoid_cusp_source_angles`) import correctly.
- No stale "Positive-parity only" references remain in surrogate_training.py.

### INS-4-001 wedge loader verification
- `_chart_from_npz` wedge branch uses `data[prefix + 'theta_to_u']` (hard KeyError) — correct, not changed to `.get()`.
- Exterior-polar and lobe branches use `data.get(prefix + 'theta_to_u')` (soft None fallback) — correct for optional field.

## Fast-tier test results
- test_lensing_surrogate.py: 123 passed
- test_lensing_surrogate_training.py: 113 passed, 90 skipped (train-tier gated)
- test_lensing_farfield_envelope.py: 36 passed, 28 skipped (train-tier gated)

## Open Issues
- INS-1-001 (carried to Librarian — doc staleness, not a code defect)

## Pre-existing (not actionable)
- INS-1-002/003 (2026-08-10): SPEC.md + DATA_CONTRACTS.yaml exterior_polar_rho_log_carrier_v1 as "ONLY known tag" — stale since V5 2D carrier shipped. Pre-existing, not this build.
