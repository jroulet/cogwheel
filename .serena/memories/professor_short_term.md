# Professor review — 2026-08-08: ExteriorPolarChart theta_to_u cusp-adapted coordinate

## Build review verdict: PASS
All 8 test files, all fast tests pass (0 failures). Engine-backed training-tier tests properly gated behind COGWHEEL_TRAIN_TIER=1.

## Key validations confirmed
1. **Numerical correctness**: theta_to_u (d^(2/3) Pearcey cusp scaling) integrated via np.interp in _evaluate_chart, mirroring wedge pattern. Served values agree with raw-theta charts within tolerance — the coordinate change is an accuracy improvement, not a model change.
2. **Load-bearing proof**: Mutation falsification tests confirm the remap is not dead code — perturbing theta_to_u measurably shifts served values.
3. **Backward compat**: theta_to_u=None falls through to raw theta_c_grid; all existing tests byte-identical.
4. **Serialization**: npz round-trip preserves theta_to_u bitwise (max|diff| = 0).
5. **Schema migration**: 'exterior_polar_rho_theta_c' hard-refused; new schema 'exterior_polar_caustic_fixed_axes' with optional theta_to_u key loads correctly.
6. **Node exactness**: grid-node served values match training within 1e-7 tolerance.
7. **Census/lobe**: Census classification correctly handles theta_to_u-bearing charts; lobe tests preserved.
8. **Training pipeline**: subdivided children propagate theta_to_u through; edge-case rejection (bounds, monotonicity) operational.

## Skipped tests (expected, deferred to operator post-build)
- BuildFarfieldPositiveParityCuspAdaptedTestCase & siblings: COGWHEEL_TRAIN_TIER=1
- Training accuracy sweeps: engine-backed, minutes-scale per class
- No concerns — all fast-gate tests pass.
