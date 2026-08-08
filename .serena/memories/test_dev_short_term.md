# Test Dev Short-Term Observations

2026-08-08: Migrated test_lensing_surrogate_lobe.py (73 tests) for WP lobe-1..4:
- theta_to_s/s_grid → theta_to_u/u_grid on all lobe test references
- _LOBE_AXIS_SCHEMA → _LOBE_AXIS_SCHEMA_NEW ('lobe_caustic_relative_v1')
- Retired V1 identity-path tests (theta_to_s=None no longer supported)
- SQRTEDGE → U_COORD: cusp-adapted u = d**(2/3) replaces sqrt-edge formula
- _engine_lobe_fixture now calls from_lobe_engine w/ cusp_angle from tile cusps
- Added 6 new test classes: CarveOutRetirement (2), LobeCuspAxisMap (10),
  CuspAdjacentRoundTrip (2), LobeSchemaHardRefuse (7), UAxisNodeExact (1),
  OpenCuspEdgeProbe (1) + 4 self-falsification classes.
- Open-cusp edge probe: smoke-scale 4x4x4 chart gives ~7% error near cusp;
  gate at 0.10 (production bar 1e-3 only at 12+ nodes). Key finding: theta_to_u
  is REQUIRED under new schema (no identity-path fallback on load).
- 63 passed, 10 skipped (golden-value tests need re-freeze).

2026-08-08: Added 30 tests across 10 new classes to test_lensing_lobe_subdivision.py:
- CarveOutRetirementTestCase (2): verifies _LOBE_CUSP_EXCLUSION_DISTANCE deleted
- LobeCuspAxisMapTestCase (10): _lobe_cusp_axis_map construction + error paths
- CuspAdjacentRoundTripTestCase (2): full serve-pipeline round-trip w/ cusp threading
- LobeSchemaHardRefuseTestCase (7): old schema tags hard-refuse, new tag validates
- UAxisNodeExactTestCase (3): B-spline at stored u-nodes reproduces envelope to 1e-7
- OpenCuspEdgeProbeTestCase (1): chart agrees w/ engine at cusp-boundary point
Plus 4 self-falsification classes. Key finding: rho_lobe must be ≤0.5 at cusp edge
for eta > DEFAULT_CAUSTIC_FLOOR=0.05; rho=0.95 at cusp edge gives eta~0.0001.
