# Inspector Short-Term Observations

## 2025-08-04: Build cusp_arm_boundary review (re-confirmed)

### Scope
WP1: Cusp arm actual boundary sweep + enable coverage constant.
- `scripts/measure_cusp_arm_actual_boundary.py` (new): Measures the
  actual accept/refuse boundary of cusp_amplification by sampling
  random source positions and finding minimum image-theta offset from
  cusp vertex where the arm serves.
- `cogwheel/lensing/surrogate.py`: `_CUSP_ARM_COVERAGE` changed from
  0.0 to 0.07 (measured, floored to 2dp conservative).
- `cogwheel/tests/test_lensing_cusp_arm_coverage.py` (new): 11 tests
  certifying the constant's value, near-vertex refusal, served-source
  coverage bound, transition monotonicity, and self-falsification.

### Findings summary
NO BUGS found. All 11 new tests PASS (5.97s). All 69 existing
surrogate tests PASS (121s). Existing census cusp-window tests PASS.
Exterior windows tests PASS (85 passed, 1 xfailed, 219s).

### Detailed analysis

#### Existing test impact of coverage=0.07:
- cusp_windows=[(0.2, 0.1)]: residual = max(0, 0.1-0.07) = 0.03.
  Tests querying theta=0.2 (exact cusp) still blocked (delta=0 < 0.03). ✓
- cusp_windows=[(-0.39, 0.05)]: residual = max(0, 0.05-0.07) = 0.0.
  Window fully absorbed. No test queries that relied on this window for blocking.
- cusp_windows=[(theta_lo, 0.02)]: residual = max(0, 0.02-0.07) = 0.0.
  Window fully absorbed. Tests avoid querying at theta_lo. ✓
- MutationFalsificationTestCase (skipped/TRAIN_TIER): cusp_windows=((0.7, 0.2)),
  query theta=0.7: residual = max(0, 0.2-0.07) = 0.13, |0.7-0.7|=0 < 0.13
  → still blocked → test logic preserved. ✓

#### API usage verified:
- cusp_amplification(w: float, source, gamma) — correct scalar w usage
- _cusp_vertex(gamma, beta=0.0, kappa=0.0, source, seed_theta, branch=1) — correct
- nearest_caustic_point(gamma, beta=0.0, source, kappa=0.0) — correct
- find_images(source, matrix) — correct
- use_pearcey_table() — correct (no-arg, returns bool)
- critical_point(gamma, 0.0, beta=0.0, kappa=0.0, branch=1) — correct
- macro_matrix(gamma, 0.0, 0.0) — correct

#### Script methodology:
- Random source sampling in disk of radius 2*max(|gamma|, 0.5)
- For each served source: finds nearest cusp vertex, then image nearest
  to vertex, computes angular offset
- Reports minimum across positive-parity configs, floors to 2dp
- Saddle parity excluded (converges to 0 due to deep-interior images)
- Refinement pass at worst-case config with N=10000

#### SPEC consistency:
- SPEC says "cusp neighborhoods are EXCLUDED (2/3-power singularity;
  served exact until the cusp fast-serving build)". This is now partially
  stale — the cusp arm IS serving (Pearcey), and the exclusion window is
  shrunk by _CUSP_ARM_COVERAGE=0.07. Librarian scope (trivial).

### Open issues:
- INS-4-001: SPEC.md cusp window description stale (Librarian scope, trivial)
- INS-2-002: SPEC.md not updated for QD ceiling (Librarian scope, trivial, carried)

### Passing test files:
- test_lensing_cusp_arm_coverage.py: 11 passed, 5.97s
- test_lensing_surrogate.py: 69 passed, 121s
- test_lensing_surrogate_census.py (targeted cusp): 2 passed, 1 skipped
- test_lensing_exterior_windows.py: 85 passed, 1 xfailed, 219s
