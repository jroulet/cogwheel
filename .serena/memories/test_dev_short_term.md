# Test Dev Short-Term Observations

2026-08-11: added 19 tests / 5 classes to test_lensing_levers.py for WP-1/WP-2:
- ConsultPearceyRefusalTestCase (4 tests): table=None => None, table inside => value/outside => None, mock pearcey => no live quadrature
- ConsultPearceyRefusalSelfFalsification (1 test): mock fires on direct pearcey call
- PearceyTableSchemaMigrationTestCase (2 tests): 0.2.0 round-trips, 0.1.0 raises ValueError
- PearceyTableCertificationTestCase.test_explicit_residual_reconstruction (1 test): P_asymp + spline resid == table.evaluate
Removed test_consult_routes_outside_box_to_live_quadrature (WP-2 killed fallback).
Updated test_corrupt_artifact_falls_back_not_serves: _consult_pearcey(None) is None.
Updated Lever-4 + PearceyTableFallbackTestCase docstrings for residual table.
2026-08-11 (build 2nd pass): added 11 tests / 3 classes:
- PearceyTableLoadFailureTestCase (4 tests): missing file → False+global None → arm refuses; warning text does NOT say 'live quadrature'
- PearceyTableLoadFailureSelfFalsification (1 test): installed table makes arm serve, not refuse
- DeadCodeDeletionGateTestCase (4 tests): git grep confirms zero def/call-site hits for demodulate/remodulate/_carrier_phase/_dominant_stationary_point in non-test cogwheel/; AST confirms absent from _pearcey_table.py; _SPLIT_BASE retained in _pearcey_cusp.py
- DeadCodeDeletionSelfFalsification (2 tests): live pearcey grep detection proves pattern has teeth; _SPLIT_BASE absent from _pearcey_table.py
Added `import warnings` at top level. Fixed _git_grep_cogwheel to use -E (extended regex) and exclude cogwheel/tests/ (avoids self-trigger of docstring mentions).

PRE-EXISTING FAILURES from coder changes: PearceyTableCertificationTestCase (3 tests, residual P-P_asymp has caustic-crossing discontinuity -> spline error 1.9e+09); LMaxSelfFalsificationTestCase (1 test, pre-existing on HEAD).

2026-08-11 (WP1 fixed-panel Gauss-Legendre overlap-band tests): added 2 tests / 2 classes to test_lensing_schwinger.py:
- OverlapBandDdMpmathAgreementTestCase (1 test): f_schwinger(60, ...) DD path vs _f_schwinger_mpmath(60, ...) cross-agreement < 5e-10 at 8 points (gamma' ∈ {0.3, 0.7, 1.3, 1.5} × y ∈ {(0.3,0.2), (0.7,0.4)}); worst measured 5.6e-11 at gamma'=1.5, y=(0.3,0.2)
- OverlapBandSelfFalsificationTestCase (1 test): mock ceil→-10 (dps=20) + relax _CERTIFICATION_TOL→100 → cross-agreement > 1e-4 proves teeth
Tolerance documented: DD path at w=60 has e^{pi*60/4} ≈ 3e20 amplification limiting dd accuracy to ~1e-10.

2026-08-11 (WP1 ppGO resolution gate self-falsification test): added 1 test to PpgoRungSelfFalsificationTestCase in test_lensing_airy_fold.py — test_resolution_gate_isolated_admit_and_refuse (4 checks, ~3.9 s). Fixture: _PPGO_SADDLE_SOURCE at gamma=1.2 (two saddle-type images, merge=None). Proves teeth by raising _PPGO_RESOLUTION_GATE→1000 (blocks at w=500 where w*delta_min=322<1000), lowering→0 (always admits), and showing w=20000 with gate=1000 still admits (w*delta_min=12880>1000). Architect's spec was wrong about w=500 being unresolved (w*delta_min≈322≫4) — the saddle source always resolves at w≥50. Key finding: _merging_fold_pair returns None for dual-saddle 2-image sources, making resolution gate the sole admission criterion.

2026-08-11 (WP1 _cusp_vertex routing fix domain tests): added 11 tests / 4 classes to test_lensing_airy_fold.py:
- InteriorCuspTableLiveAgreementTestCase (4 tests): interior cusp source serves via both table and live quadrature; route is 'pearcey'; table-live agreement to 1e-5 relative; sweep w∈[20,80] diagnostic plot
- CuspVertexSourceDistanceSelectionTestCase (3 tests): _cusp_vertex returns source-plane-closest astroid cusp; seed_theta independence; Voronoi partition diagnostic plot
- ExteriorPpgoUnaffectedTestCase (2 tests): exterior ppGO route unchanged by routing fix; values finite and deterministic with/without table
- InteriorCuspSelfFalsificationTestCase (2 tests): cleared-table still serves; corrupted _cusp_vertex violates distance gate
Fixture uses _CUSP_FIXTURES[0] = (0.5, 0.20, 0.25π) — interior 4-image source with cusp cluster calibration passing. Table-live diff ~1e-7 (relative ~1e-8), gated at 1e-5.
PRE-EXISTING FAILURES (not caused by these tests): 8 vertex-related tests fail due to new _cusp_vertex returning finite wedge-tip vertex where old code returned None at wedge-edge configs — the coder's WP1 behavior change re: multi-candidate source-distance selection.
