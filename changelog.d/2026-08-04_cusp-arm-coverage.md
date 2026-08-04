## 2026-08-04

### Lensing surrogate: cusp arm coverage enabled (_CUSP_ARM_COVERAGE = 0.07 rad)

The `_CUSP_ARM_COVERAGE` constant in `cogwheel/lensing/surrogate.py` is now
set to `0.07 rad`, enabling the Pearcey arm to serve draws within that
angular distance of each cusp vertex.

**Background:** the surrogate's `select_chart` previously excluded a
generous cusp window from tube charts and served those draws with the exact
quadrature engine.  The new constant was derived by sweeping
`delta_theta` from each cusp vertex across `gamma = [0.1..1.5]`,
`w = [10..40]` and reading the minimum angle at which
`cusp_amplification` actually *accepts* a node (measured by
`scripts/measure_cusp_arm_actual_boundary.py`).  The minimum measured
boundary is 0.07 rad across all configurations.

With the Pearcey table artifact (shipped in commit `c715bcd`) registered
and `_CUSP_ARM_COVERAGE = 0.07 rad`, draws within 0.07 rad of the cusp
vertex are now served by the certified Pearcey arm instead of the exact
engine.  The residual exclusion window beyond the arm's certified reach
continues to fall through to the exact engine.

**New files:**
- `cogwheel/tests/test_lensing_cusp_arm_coverage.py` (393 lines)
- `scripts/measure_cusp_arm_actual_boundary.py` (239 lines)

Commit: `ddd8980`
